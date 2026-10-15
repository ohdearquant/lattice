use super::*;
use syn::spanned::Spanned;

pub(super) const TARGETS: &[&str] = &[
    "examples/layer_sweep.rs",
    "examples/dump_embeddinggemma2_vision.rs",
    "src/bin/bench_gdn_prefill_ab.rs",
    "src/bin/bench_vision_prefill_ab.rs",
    "src/bin/eval_perplexity.rs",
];

const WORK_NAMES: &[&str] = &[
    "encode_all_layers",
    "encode_parts_at_widths_metal",
    "forward_prefill_gdn_isolated_chunk",
    "forward_prefill_production_chunk",
    "run_multimodal_prefill",
    "load_lora_adapter",
    "compute_perplexity",
];

#[derive(Default)]
struct Functions<'a>(Vec<&'a ItemFn>);

impl<'a> Visit<'a> for Functions<'a> {
    fn visit_item_fn(&mut self, function: &'a ItemFn) {
        self.0.push(function);
        syn::visit::visit_item_fn(self, function);
    }
}

#[derive(Default)]
struct Body<'a> {
    deferred: bool,
    calls: Vec<(&'a syn::ExprCall, bool)>,
    methods: Vec<(&'a syn::ExprMethodCall, bool)>,
    locals: Vec<(&'a syn::Local, bool)>,
    paths: Vec<&'a syn::ExprPath>,
    bindings: Vec<(&'a syn::PatIdent, bool)>,
    assignments: Vec<&'a syn::ExprAssign>,
    macros: Vec<&'a syn::Macro>,
}

impl<'a> Visit<'a> for Body<'a> {
    fn visit_expr_call(&mut self, call: &'a syn::ExprCall) {
        self.calls.push((call, self.deferred));
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'a syn::ExprMethodCall) {
        self.methods.push((call, self.deferred));
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_local(&mut self, local: &'a syn::Local) {
        self.locals.push((local, self.deferred));
        syn::visit::visit_local(self, local);
    }

    fn visit_expr_path(&mut self, path: &'a syn::ExprPath) {
        self.paths.push(path);
        syn::visit::visit_expr_path(self, path);
    }

    fn visit_pat_ident(&mut self, binding: &'a syn::PatIdent) {
        self.bindings.push((binding, self.deferred));
        syn::visit::visit_pat_ident(self, binding);
    }

    fn visit_expr_assign(&mut self, assignment: &'a syn::ExprAssign) {
        self.assignments.push(assignment);
        syn::visit::visit_expr_assign(self, assignment);
    }

    fn visit_macro(&mut self, mac: &'a syn::Macro) {
        self.macros.push(mac);
        syn::visit::visit_macro(self, mac);
    }

    fn visit_expr_closure(&mut self, closure: &'a syn::ExprClosure) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_expr_closure(self, closure);
        self.deferred = previous;
    }

    fn visit_expr_async(&mut self, expression: &'a syn::ExprAsync) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_expr_async(self, expression);
        self.deferred = previous;
    }

    fn visit_item(&mut self, item: &'a syn::Item) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_item(self, item);
        self.deferred = previous;
    }
}

fn offset(node: &impl Spanned) -> usize {
    node.span().byte_range().start
}

fn token(parsed: &StructuredSource, offset: usize) -> Result<usize, String> {
    parsed
        .tokens
        .iter()
        .position(|token| token.offset == offset)
        .ok_or_else(|| format!("{}: unmapped AST byte {offset}", parsed.context))
}

fn live(parsed: &StructuredSource, offset: usize) -> Result<bool, String> {
    parsed.formula_at(token(parsed, offset)?)?.satisfiable()
}

fn bare(expression: &syn::Expr, name: &str) -> bool {
    matches!(expression, syn::Expr::Path(path) if path.attrs.is_empty()
        && path.qself.is_none() && path.path.is_ident(name))
}

fn call_path(call: &syn::ExprCall, names: &[&str]) -> bool {
    let syn::Expr::Path(path) = &*call.func else {
        return false;
    };
    path.qself.is_none()
        && path.attrs.is_empty()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == names.len()
        && path.path.segments.iter().zip(names).all(|(segment, name)| {
            segment.ident == *name && matches!(segment.arguments, syn::PathArguments::None)
        })
}

fn callee_offset(call: &syn::ExprCall) -> Result<usize, String> {
    let syn::Expr::Path(path) = &*call.func else {
        return Err("indirect callee".into());
    };
    path.path
        .segments
        .last()
        .map(|segment| offset(&segment.ident))
        .ok_or_else(|| "empty callee".into())
}

fn mut_borrow(expression: &syn::Expr, name: &str) -> bool {
    matches!(expression, syn::Expr::Reference(reference) if reference.mutability.is_some()
        && reference.attrs.is_empty() && bare(&reference.expr, name))
}

fn local_name(local: &syn::Local) -> Option<&syn::Ident> {
    let pattern = match &local.pat {
        syn::Pat::Type(typed) => &*typed.pat,
        pattern => pattern,
    };
    match pattern {
        syn::Pat::Ident(binding) if binding.subpat.is_none() && binding.by_ref.is_none() => {
            Some(&binding.ident)
        }
        _ => None,
    }
}

fn unique_local<'a>(body: &Body<'a>, name: &str) -> Result<&'a syn::Local, String> {
    let locals = body
        .locals
        .iter()
        .filter(|(local, _)| local_name(local).is_some_and(|ident| ident == name))
        .collect::<Vec<_>>();
    if locals.len() != 1 || locals[0].1 {
        return Err(format!("missing, shadowed, or deferred local {name}"));
    }
    Ok(locals[0].0)
}

fn initializer(local: &syn::Local) -> Result<&syn::Expr, String> {
    local
        .init
        .as_ref()
        .filter(|init| init.diverge.is_none())
        .map(|init| &*init.expr)
        .ok_or_else(|| "expected a direct local initializer".into())
}

fn synchronous(function: &ItemFn) -> Result<(), String> {
    if function.sig.asyncness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
    {
        return Err(format!(
            "{} is no longer an ordinary synchronous function",
            function.sig.ident
        ));
    }
    Ok(())
}

fn function<'a>(
    parsed: &StructuredSource,
    functions: &Functions<'a>,
    name: &str,
) -> Result<&'a ItemFn, String> {
    let mut found = Vec::new();
    for function in &functions.0 {
        let enclosing = parsed.formula_at(token(parsed, offset(&function.sig.ident))?)?;
        let own = syn_attributes_formula(&function.attrs, false, &parsed.context)?;
        if function.sig.ident == name && CfgFormula::all([enclosing, own]).satisfiable()? {
            found.push(*function);
        }
    }
    if found.len() != 1 {
        return Err(format!("expected one active function {name}"));
    }
    synchronous(found[0])?;
    Ok(found[0])
}

fn body(function: &ItemFn) -> Body<'_> {
    let mut body = Body::default();
    body.visit_block(&function.block);
    body
}

fn closure_body(closure: &syn::ExprClosure) -> Result<Body<'_>, String> {
    if closure.asyncness.is_some() || closure.movability.is_some() || closure.capture.is_some() {
        return Err("expected a nonescaping synchronous borrowed closure".into());
    }
    let mut body = Body::default();
    body.visit_expr(&closure.body);
    Ok(body)
}

fn exact_identifiers(
    parsed: &StructuredSource,
    name: &str,
    expected: &[usize],
) -> Result<(), String> {
    let actual = parsed
        .tokens
        .iter()
        .filter(|token| token.ident() == Some(name))
        .map(|token| token.offset)
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "{name} has an unreviewed alias, use, definition, or call"
        ));
    }
    Ok(())
}

fn syntax_shape(source: &str) -> Result<Vec<String>, String> {
    let tokens = rust_tokens(source)?;
    Ok(tokens
        .iter()
        .enumerate()
        .filter(|(index, token)| {
            !token.is_punct(',')
                || !tokens.get(index + 1).is_some_and(|next| {
                    next.is_punct(')')
                        || next.is_punct(']')
                        || next.is_punct('>')
                        || next.is_punct('}')
                })
        })
        .map(|(_, token)| format!("{:?}", token.kind))
        .collect())
}

fn exact_syntax(source: &str, node: &impl Spanned, expected: &str) -> Result<(), String> {
    let actual = source
        .get(node.span().byte_range())
        .ok_or("signature span outside source")?;
    if syntax_shape(actual)? != syntax_shape(expected)? {
        return Err(format!(
            "reviewed synchronous signature/import changed: expected {expected}"
        ));
    }
    Ok(())
}

fn validate_bench_import(
    parsed: &StructuredSource,
    owner: &ItemFn,
    grouped: bool,
) -> Result<(), String> {
    let expected = if grouped {
        "use lattice_inference::forward::metal_qwen35::bench_support::{self, GdnIsolatedChunkTiming};"
    } else {
        "use lattice_inference::forward::metal_qwen35::bench_support;"
    };
    let imports = owner
        .block
        .stmts
        .iter()
        .filter_map(|statement| match statement {
            syn::Stmt::Item(syn::Item::Use(import)) => Some(import),
            _ => None,
        })
        .filter(|import| exact_syntax(&parsed.source, *import, expected).is_ok())
        .collect::<Vec<_>>();
    if imports.len() != 1 {
        return Err("missing exact library bench_support import in run".into());
    }
    let import_range = imports[0].span().byte_range();
    let mut namespace_uses = parsed
        .tokens
        .iter()
        .filter(|token| {
            import_range.contains(&token.offset) && token.ident() == Some("bench_support")
        })
        .map(|token| token.offset)
        .collect::<Vec<_>>();
    let outer = body(owner);
    for (call, _) in outer.calls {
        if let syn::Expr::Path(path) = &*call.func
            && path.qself.is_none()
            && path.path.leading_colon.is_none()
            && path.path.segments.len() == 2
            && path.path.segments[0].ident == "bench_support"
        {
            namespace_uses.push(offset(&path.path.segments[0].ident));
        }
    }
    exact_identifiers(parsed, "bench_support", &namespace_uses)?;
    validate_crate_namespace(parsed)
}

fn validate_crate_namespace(parsed: &StructuredSource) -> Result<(), String> {
    #[derive(Default)]
    struct Imports<'a>(Vec<&'a syn::ItemUse>);
    impl<'a> Visit<'a> for Imports<'a> {
        fn visit_item_use(&mut self, import: &'a syn::ItemUse) {
            self.0.push(import);
        }
    }
    fn replaces_crate(tree: &syn::UseTree) -> bool {
        match tree {
            syn::UseTree::Path(path) => replaces_crate(&path.tree),
            syn::UseTree::Name(name) => name.ident == "lattice_inference",
            syn::UseTree::Rename(rename) => rename.rename == "lattice_inference",
            syn::UseTree::Group(group) => group.items.iter().any(replaces_crate),
            syn::UseTree::Glob(_) => true,
        }
    }
    let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
    let mut all_imports = Imports::default();
    all_imports.visit_file(&syntax);
    for import in all_imports.0 {
        if live(parsed, import.use_token.span.byte_range().start)? && replaces_crate(&import.tree) {
            return Err("active import may replace the lattice_inference namespace".into());
        }
    }
    for (index, token) in parsed.tokens.iter().enumerate() {
        if token.ident() == Some("lattice_inference") {
            let previous = index
                .checked_sub(1)
                .and_then(|index| parsed.tokens.get(index))
                .and_then(RustToken::ident);
            if matches!(
                previous,
                Some("mod" | "as" | "crate" | "type" | "struct" | "enum" | "trait")
            ) {
                return Err("lattice_inference namespace is locally replaced".into());
            }
        }
    }
    Ok(())
}

fn guarded(parsed: &StructuredSource, function: &ItemFn, offsets: &[usize]) -> Result<(), String> {
    let sites = offsets
        .iter()
        .map(|offset| token(parsed, *offset))
        .collect::<Result<Vec<_>, _>>()?;
    let close = token(
        parsed,
        function.block.brace_token.span.close().byte_range().start,
    )?;
    for site in &sites {
        if parsed.enclosing_function_path(*site) != function.sig.ident.to_string()
            || !parsed.formula_at(*site)?.satisfiable()?
        {
            return Err("work moved to a different or inactive function".into());
        }
    }
    parsed.validate_work_sites(
        &sites,
        SHARED_LOCK_SELECTOR,
        GuardRequirement::Function {
            closing_brace: close,
        },
    )
}

fn state_parameter(pattern: &syn::Pat) -> bool {
    let syn::Pat::Type(typed) = pattern else {
        return false;
    };
    matches!(&*typed.pat, syn::Pat::Ident(binding) if binding.ident == "state" && binding.subpat.is_none())
        && matches!(&*typed.ty, syn::Type::Reference(reference) if reference.mutability.is_some()
            && matches!(&*reference.elem, syn::Type::Path(path) if path.path.is_ident("MetalQwen35State")))
}

fn function_state_parameter(function: &ItemFn) -> Result<(), String> {
    let Some(syn::FnArg::Typed(typed)) = function.sig.inputs.first() else {
        return Err("missing state parameter".into());
    };
    if !state_parameter(&syn::Pat::Type(typed.clone())) {
        return Err("state parameter contract changed".into());
    }
    Ok(())
}

fn state_uses(body: &Body<'_>, allowed: &[usize]) -> Result<(), String> {
    for mac in &body.macros {
        if rust_tokens(&mac.tokens.to_string())?
            .iter()
            .any(|token| token.ident() == Some("state"))
        {
            return Err("state use inside an opaque macro is not synchronous evidence".into());
        }
    }
    let actual = body
        .paths
        .iter()
        .filter(|path| path.path.is_ident("state"))
        .map(|path| offset(*path))
        .collect::<BTreeSet<_>>();
    if actual != allowed.iter().copied().collect() {
        return Err("state is aliased or escapes the reviewed synchronous calls".into());
    }
    if body
        .bindings
        .iter()
        .any(|(binding, _)| binding.ident == "state")
    {
        return Err("state parameter is rebound".into());
    }
    Ok(())
}

fn methods(
    body: &Body<'_>,
    method: &str,
    receiver: &str,
    count: usize,
) -> Result<Vec<usize>, String> {
    let calls = body
        .methods
        .iter()
        .filter(|(call, _)| call.method == method)
        .collect::<Vec<_>>();
    if calls.len() != count
        || calls
            .iter()
            .any(|(call, deferred)| *deferred || !bare(&call.receiver, receiver))
    {
        return Err(format!(
            "{receiver}.{method} count, receiver, or synchronous shape changed"
        ));
    }
    Ok(calls.iter().map(|(call, _)| offset(&call.method)).collect())
}

fn work_census(parsed: &StructuredSource, claimed: &[usize]) -> Result<(), String> {
    let actual = parsed
        .tokens
        .iter()
        .filter(|token| token.ident().is_some_and(|name| WORK_NAMES.contains(&name)))
        .map(|token| token.offset)
        .collect::<BTreeSet<_>>();
    if actual != claimed.iter().copied().collect() {
        return Err("unclassified indirect work occurrence".into());
    }
    for site in claimed {
        if !live(parsed, *site)? {
            return Err("inactive protected work is not evidence".into());
        }
    }
    Ok(())
}

fn direct_calls<'a>(
    body: &Body<'a>,
    name: &str,
    count: usize,
) -> Result<Vec<&'a syn::ExprCall>, String> {
    let calls = body
        .calls
        .iter()
        .filter(|(call, _)| call_path(call, &[name]))
        .collect::<Vec<_>>();
    if calls.len() != count || calls.iter().any(|(_, deferred)| *deferred) {
        return Err(format!(
            "{name} direct synchronous invocation count changed"
        ));
    }
    Ok(calls.into_iter().map(|(call, _)| *call).collect())
}

fn validate_local_closures(
    parsed: &StructuredSource,
    functions: &Functions<'_>,
    specs: &[(&str, &str)],
) -> Result<(), String> {
    let owner = function(parsed, functions, "run")?;
    validate_bench_import(parsed, owner, specs.len() == 2)?;
    let outer = body(owner);
    let mut claimed = Vec::new();
    let mut invocations = Vec::new();
    for (name, helper) in specs {
        let local = unique_local(&outer, name)?;
        let syn::Expr::Closure(closure) = initializer(local)? else {
            return Err("expected local closure".into());
        };
        if !closure.inputs.first().is_some_and(state_parameter) {
            return Err("closure state parameter changed".into());
        }
        let inner = closure_body(closure)?;
        let calls = inner
            .calls
            .iter()
            .filter(|(call, _)| call_path(call, &["bench_support", helper]))
            .collect::<Vec<_>>();
        if calls.len() != 1
            || calls[0].1
            || !calls[0]
                .0
                .args
                .first()
                .is_some_and(|arg| bare(arg, "state"))
        {
            return Err(format!(
                "{name} must synchronously call bench_support::{helper}(state, ...)"
            ));
        }
        let mut allowed_state = vec![offset(&calls[0].0.args[0])];
        for (method, deferred) in &inner.methods {
            if bare(&method.receiver, "state") {
                if *deferred || method.method != "reset_state" || !method.args.is_empty() {
                    return Err("unreviewed state method inside local closure".into());
                }
                allowed_state.push(offset(&method.receiver));
            }
        }
        state_uses(&inner, &allowed_state)?;
        claimed.push(callee_offset(calls[0].0)?);
        let calls = direct_calls(&outer, name, 4)?;
        let mut uses = vec![offset(local_name(local).ok_or("missing closure binding")?)];
        for call in calls {
            if !call
                .args
                .first()
                .is_some_and(|arg| mut_borrow(arg, "state"))
            {
                return Err("local closure must borrow its owner's state".into());
            }
            let site = callee_offset(call)?;
            if site <= offset(local) {
                return Err("closure invocation precedes its definition".into());
            }
            uses.push(site);
            invocations.push(site);
        }
        exact_identifiers(parsed, name, &uses)?;
    }
    guarded(parsed, owner, &invocations)?;
    work_census(parsed, &claimed)
}

fn vec_infer(local: &syn::Local) -> bool {
    let syn::Pat::Type(typed) = &local.pat else {
        return false;
    };
    let syn::Type::Path(path) = &*typed.ty else {
        return false;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return false;
    }
    let segment = &path.path.segments[0];
    matches!(&segment.arguments, syn::PathArguments::AngleBracketed(arguments)
        if segment.ident == "Vec" && arguments.args.len() == 1
        && matches!(arguments.args.first(), Some(syn::GenericArgument::Type(syn::Type::Infer(_)))))
}

fn validate_iterators(parsed: &StructuredSource, functions: &Functions<'_>) -> Result<(), String> {
    let owner = function(parsed, functions, "main")?;
    let outer = body(owner);
    let mut claimed = Vec::new();
    let mut anchors = Vec::new();
    let mut vec_types = Vec::new();
    for (local, array, count) in [
        ("all_embeddings", "test_texts", 1),
        ("pair_embeddings", "pairs", 3),
    ] {
        let collection = unique_local(&outer, local)?;
        if !vec_infer(collection) {
            return Err("map must collect immediately into Vec<_>".into());
        }
        let syn::Pat::Type(typed) = &collection.pat else {
            unreachable!()
        };
        vec_types.push(offset(&typed.ty));
        let syn::Expr::MethodCall(collect) = initializer(collection)? else {
            return Err("missing eager collect".into());
        };
        let syn::Expr::MethodCall(map) = &*collect.receiver else {
            return Err("collect must consume map directly".into());
        };
        let syn::Expr::MethodCall(iter) = &*map.receiver else {
            return Err("map must consume the reviewed array iterator".into());
        };
        if collect.method != "collect"
            || !collect.args.is_empty()
            || collect.turbofish.is_some()
            || map.method != "map"
            || map.args.len() != 1
            || map.turbofish.is_some()
            || iter.method != "iter"
            || !iter.args.is_empty()
            || iter.turbofish.is_some()
            || !bare(&iter.receiver, array)
        {
            return Err("eager array.iter().map(...).collect() shape changed".into());
        }
        let array_local = unique_local(&outer, array)?;
        if !matches!(initializer(array_local)?, syn::Expr::Array(_)) {
            return Err("reviewed iterator input is no longer an array".into());
        }
        if outer
            .assignments
            .iter()
            .any(|assignment| bare(&assignment.left, array))
        {
            return Err("iterator input was reassigned".into());
        }
        let syn::Expr::Closure(closure) = &map.args[0] else {
            return Err("map callback is not inline".into());
        };
        claimed.extend(methods(
            &closure_body(closure)?,
            "encode_all_layers",
            "model",
            count,
        )?);
        anchors.push(offset(&collect.method));
    }
    exact_identifiers(parsed, "Vec", &vec_types)?;
    let direct = outer
        .methods
        .iter()
        .filter(|(call, deferred)| call.method == "encode_all_layers" && !*deferred)
        .collect::<Vec<_>>();
    if direct.len() != 1 || !bare(&direct[0].0.receiver, "model") {
        return Err("direct all-layer timing call changed".into());
    }
    claimed.push(offset(&direct[0].0.method));
    anchors.push(offset(&direct[0].0.method));
    guarded(parsed, owner, &anchors)?;
    work_census(parsed, &claimed)
}

fn validate_callback(
    parsed: &StructuredSource,
    functions: &Functions<'_>,
    syntax: &syn::File,
) -> Result<(), String> {
    let helper = function(parsed, functions, "dump")?;
    exact_syntax(
        &parsed.source,
        &helper.sig,
        "fn dump(args: &Args, model: &EmbeddingGemma2Model, forward: &mut Forward<'_>) -> Result<(), Box<dyn std::error::Error>>",
    )?;
    let Some(syn::FnArg::Typed(parameter)) = helper.sig.inputs.iter().nth(2) else {
        return Err("missing forward callback parameter".into());
    };
    let syn::Pat::Ident(binding) = &*parameter.pat else {
        return Err("callback parameter pattern changed".into());
    };
    if binding.ident != "forward"
        || !matches!(&*parameter.ty, syn::Type::Reference(reference)
        if reference.mutability.is_some() && matches!(&*reference.elem, syn::Type::Path(path)
            if path.path.segments.len() == 1 && path.path.segments[0].ident == "Forward"))
    {
        return Err("dump must borrow Forward mutably".into());
    }
    let aliases = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Type(alias) if alias.ident == "Forward" => Some(alias),
            _ => None,
        })
        .collect::<Vec<_>>();
    if aliases.len() != 1 {
        return Err("missing callback type alias".into());
    }
    exact_syntax(&parsed.source, &aliases[0].generics, "<'a>")?;
    exact_syntax(
        &parsed.source,
        &aliases[0].ty,
        "dyn FnMut(&[EmbeddingGemma2Part], &[usize]) -> Result<(EmbeddingGemma2PartsEncoding, Option<VisionTimings>), Box<dyn std::error::Error>> + 'a",
    )?;
    if aliases[0]
        .attrs
        .iter()
        .any(|attribute| !attribute.path().is_ident("doc"))
    {
        return Err("Forward alias has an unreviewed conditional or transforming attribute".into());
    }
    let syn::Type::Reference(reference) = &*parameter.ty else {
        unreachable!()
    };
    let syn::Type::Path(forward_type) = &*reference.elem else {
        unreachable!()
    };
    exact_identifiers(
        parsed,
        "Forward",
        &[
            offset(&aliases[0].ident),
            offset(&forward_type.path.segments[0].ident),
        ],
    )?;
    let syn::Type::TraitObject(object) = &*aliases[0].ty else {
        return Err("Forward is no longer a synchronous FnMut trait object".into());
    };
    let traits = object
        .bounds
        .iter()
        .filter_map(|bound| match bound {
            syn::TypeParamBound::Trait(bound) => Some(bound),
            _ => None,
        })
        .collect::<Vec<_>>();
    if traits.len() != 1
        || traits[0].path.segments.len() != 1
        || traits[0].path.segments[0].ident != "FnMut"
    {
        return Err("Forward must remain FnMut".into());
    }
    let inner = body(helper);
    let invokes = direct_calls(&inner, "forward", 1)?;
    exact_identifiers(
        parsed,
        "forward",
        &[offset(&binding.ident), callee_offset(invokes[0])?],
    )?;
    let mut helper_uses = vec![offset(&helper.sig.ident)];
    let mut claimed = Vec::new();
    for (owner_name, metal) in [("main", false), ("dump_metal", true)] {
        let owner = function(parsed, functions, owner_name)?;
        let outer = body(owner);
        let calls = direct_calls(&outer, "dump", 1)?;
        let call = calls[0];
        if call.args.len() != 3 {
            return Err("dump callback argument position changed".into());
        }
        let syn::Expr::Reference(reference) = &call.args[2] else {
            return Err("dump callback must be borrowed inline".into());
        };
        let syn::Expr::Closure(closure) = &*reference.expr else {
            return Err("dump callback must be inline".into());
        };
        if reference.mutability.is_none() {
            return Err("dump callback must be mutable".into());
        }
        let callback = closure_body(closure)?;
        if metal {
            claimed.extend(methods(
                &callback,
                "encode_parts_at_widths_metal",
                "model",
                1,
            )?);
            let (method, _) = callback
                .methods
                .iter()
                .find(|(method, _)| method.method == "encode_parts_at_widths_metal")
                .ok_or("missing Metal call")?;
            if !method
                .args
                .first()
                .is_some_and(|arg| mut_borrow(arg, "state"))
            {
                return Err("Metal callback must borrow state".into());
            }
            guarded(parsed, owner, &[callee_offset(call)?])?;
        } else {
            methods(&callback, "encode_parts_at_widths", "model", 1)?;
        }
        helper_uses.push(callee_offset(call)?);
    }
    exact_identifiers(parsed, "dump", &helper_uses)?;
    work_census(parsed, &claimed)
}

fn validate_helper(parsed: &StructuredSource, functions: &Functions<'_>) -> Result<(), String> {
    let helper = function(parsed, functions, "run_on_metal_state")?;
    function_state_parameter(helper)?;
    let inner = body(helper);
    let mut claimed = methods(&inner, "load_lora_adapter", "state", 1)?;
    claimed.extend(methods(&inner, "compute_perplexity", "state", 1)?);
    let state_receivers = inner
        .methods
        .iter()
        .filter(|(method, _)| bare(&method.receiver, "state"))
        .map(|(method, _)| offset(&method.receiver))
        .collect::<Vec<_>>();
    if state_receivers.len() != 2 {
        return Err("unreviewed helper state method".into());
    }
    state_uses(&inner, &state_receivers)?;
    let mut uses = vec![offset(&helper.sig.ident)];
    for owner_name in ["run_metal_q4", "run_metal_model_dir"] {
        let owner = function(parsed, functions, owner_name)?;
        let outer = body(owner);
        let calls = direct_calls(&outer, "run_on_metal_state", 1)?;
        if !calls[0]
            .args
            .first()
            .is_some_and(|arg| mut_borrow(arg, "state"))
        {
            return Err("helper must borrow caller state".into());
        }
        let site = callee_offset(calls[0])?;
        guarded(parsed, owner, &[site])?;
        uses.push(site);
    }
    exact_identifiers(parsed, "run_on_metal_state", &uses)?;
    let cpu = function(parsed, functions, "main")?;
    validate_cpu_perplexity_receiver(parsed, functions, cpu)?;
    claimed.extend(methods(&body(cpu), "compute_perplexity", "model", 1)?);
    work_census(parsed, &claimed)
}

fn validate_cpu_perplexity_receiver(
    parsed: &StructuredSource,
    functions: &Functions<'_>,
    cpu: &ItemFn,
) -> Result<(), String> {
    validate_crate_namespace(parsed)?;
    let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
    let imports = syntax.items.iter().filter_map(|item| match item {
        syn::Item::Use(import) => Some(import), _ => None,
    }).filter(|import| exact_syntax(&parsed.source, *import,
        "use lattice_inference::model::qwen35::{PerplexityConfig, PerplexityReport, Qwen35Model};").is_ok()).collect::<Vec<_>>();
    if imports.len() != 1 {
        return Err("CPU perplexity requires the reviewed Qwen35Model import".into());
    }
    let import_range = imports[0].span().byte_range();
    let mut model_type_uses = parsed
        .tokens
        .iter()
        .filter(|token| {
            import_range.contains(&token.offset) && token.ident() == Some("Qwen35Model")
        })
        .map(|token| token.offset)
        .collect::<Vec<_>>();
    for function in &functions.0 {
        for (call, _) in body(function).calls {
            if call_path(call, &["Qwen35Model", "from_safetensors"])
                && let syn::Expr::Path(path) = &*call.func
            {
                model_type_uses.push(offset(&path.path.segments[0].ident));
            }
        }
    }
    exact_identifiers(parsed, "Qwen35Model", &model_type_uses)?;
    let outer = body(cpu);
    let local = unique_local(&outer, "model")?;
    if outer
        .bindings
        .iter()
        .filter(|(binding, _)| binding.ident == "model")
        .count()
        != 1
        || outer
            .assignments
            .iter()
            .any(|assignment| bare(&assignment.left, "model"))
    {
        return Err("CPU perplexity receiver was rebound".into());
    }
    exact_syntax(
        &parsed.source,
        initializer(local)?,
        r#"match Qwen35Model::from_safetensors(&model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: failed to load model: {e}");
            return ExitCode::FAILURE;
        }
    }"#,
    )
}

fn validate_source(target: &str, parsed: &StructuredSource) -> Result<(), String> {
    let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
    let mut functions = Functions::default();
    functions.visit_file(&syntax);
    match target {
        "examples/layer_sweep.rs" => validate_iterators(parsed, &functions),
        "examples/dump_embeddinggemma2_vision.rs" => validate_callback(parsed, &functions, &syntax),
        "src/bin/bench_gdn_prefill_ab.rs" => validate_local_closures(
            parsed,
            &functions,
            &[
                ("run_once", "forward_prefill_gdn_isolated_chunk"),
                ("run_once_production", "forward_prefill_production_chunk"),
            ],
        ),
        "src/bin/bench_vision_prefill_ab.rs" => {
            validate_local_closures(parsed, &functions, &[("run_arm", "run_multimodal_prefill")])
        }
        "src/bin/eval_perplexity.rs" => validate_helper(parsed, &functions),
        _ => Err(format!("no indirect measurement recipe for {target}")),
    }
}

fn validate_library_source(
    source: &str,
    helper: &str,
    expected_methods: &[&str],
) -> Result<(), String> {
    let syntax = syn::parse_file(source).map_err(|reason| reason.to_string())?;
    let mut functions = Functions::default();
    functions.visit_file(&syntax);
    let found = functions
        .0
        .iter()
        .filter(|function| function.sig.ident == helper)
        .collect::<Vec<_>>();
    if found.len() != 1 {
        return Err(format!("trusted helper {helper} moved or disappeared"));
    }
    let function = found[0];
    let signature = match helper {
        "forward_prefill_gdn_isolated_chunk" => {
            "fn forward_prefill_gdn_isolated_chunk(state: &mut MetalQwen35State, token_ids: &[u32], start_pos: usize) -> Result<GdnIsolatedChunkTiming, crate::error::InferenceError>"
        }
        "forward_prefill_production_chunk" => {
            "fn forward_prefill_production_chunk(state: &mut MetalQwen35State, token_ids: &[u32], start_pos: usize) -> Result<(), crate::error::InferenceError>"
        }
        "run_multimodal_prefill" => {
            "fn run_multimodal_prefill(state: &mut MetalQwen35State, request: &crate::vision::multimodal::Qwen35VisionRequest, cfg: &Qwen35Config, always_emit_head: bool) -> Result<Vec<f32>, String>"
        }
        _ => return Err("unknown trusted helper".into()),
    };
    exact_syntax(source, &function.sig, signature)?;
    let inner_modules = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Mod(module) if module.ident == "inner" => Some(module),
            _ => None,
        })
        .collect::<Vec<_>>();
    if inner_modules.len() != 1 {
        return Err("trusted helper requires one inner module".into());
    }
    let Some((_, inner_items)) = &inner_modules[0].content else {
        return Err("trusted inner module moved out of file".into());
    };
    let bench_modules = inner_items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Mod(module) if module.ident == "bench_support" => Some(module),
            _ => None,
        })
        .collect::<Vec<_>>();
    if bench_modules.len() != 1 || !matches!(bench_modules[0].vis, syn::Visibility::Public(_)) {
        return Err("trusted helper requires public inner::bench_support".into());
    }
    let Some((_, bench_items)) = &bench_modules[0].content else {
        return Err("trusted bench_support moved out of file".into());
    };
    if !bench_items.iter().any(|item| matches!(item, syn::Item::Fn(candidate)
        if offset(&candidate.sig.ident) == offset(&function.sig.ident) && matches!(candidate.vis, syn::Visibility::Public(_)))) {
        return Err("trusted helper moved outside inner::bench_support".into());
    }
    let reexports = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Use(import) if matches!(import.vis, syn::Visibility::Public(_)) => {
                Some(import)
            }
            _ => None,
        })
        .filter(|import| exact_syntax(source, &import.tree, "inner::bench_support").is_ok())
        .collect::<Vec<_>>();
    if reexports.len() != 1 {
        return Err("trusted bench_support public re-export changed".into());
    }
    let active = CfgFormula::Atom("feature=bench-internals".into());
    for attributes in [
        &inner_modules[0].attrs,
        &bench_modules[0].attrs,
        &function.attrs,
        &reexports[0].attrs,
    ] {
        if !active.implies(&syn_attributes_formula(
            attributes,
            false,
            "trusted bench helper",
        )?)? {
            return Err(
                "trusted helper namespace is not available throughout the benchmark configuration"
                    .into(),
            );
        }
    }
    synchronous(function)?;
    function_state_parameter(function)?;
    let inner = body(function);
    let mut allowed_state = Vec::new();
    for method in expected_methods {
        methods(&inner, method, "state", 1)?;
    }
    for (method, deferred) in &inner.methods {
        if bare(&method.receiver, "state") {
            if *deferred || !expected_methods.contains(&method.method.to_string().as_str()) {
                return Err("trusted helper state method changed".into());
            }
            allowed_state.push(offset(&method.receiver));
        }
    }
    for (call, deferred) in &inner.calls {
        if call_path(call, &["preflight_bench_prefill"]) {
            if *deferred || !call.args.first().is_some_and(|arg| bare(arg, "state")) {
                return Err("trusted preflight shape changed".into());
            }
            allowed_state.push(offset(&call.args[0]));
        }
    }
    state_uses(&inner, &allowed_state)
}

pub(super) fn validate_target(target: &str, sources: &[ParsedModuleSource]) -> Result<(), String> {
    if !TARGETS.contains(&target) {
        return Err(format!("unknown indirect target {target}"));
    }
    let roots = sources
        .iter()
        .filter(|source| source.relative == target)
        .collect::<Vec<_>>();
    if roots.len() != 1 {
        return Err("indirect root missing or duplicated".into());
    }
    for source in sources {
        if source.relative != target
            && source
                .parsed
                .tokens
                .iter()
                .any(|token| token.ident().is_some_and(|name| WORK_NAMES.contains(&name)))
        {
            return Err("indirect work moved into an unreviewed module".into());
        }
    }
    validate_source(target, &roots[0].parsed)?;
    let helpers: &[(&str, &[&str])] = match target {
        "src/bin/bench_gdn_prefill_ab.rs" => &[
            (
                "forward_prefill_gdn_isolated_chunk",
                &["forward_prefill_chunk_gdn_isolated"],
            ),
            (
                "forward_prefill_production_chunk",
                &["forward_prefill_batched_chunk"],
            ),
        ],
        "src/bin/bench_vision_prefill_ab.rs" => &[(
            "run_multimodal_prefill",
            &["forward_step_injected_mrope", "forward_step_mrope"],
        )],
        _ => &[],
    };
    if !helpers.is_empty() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/forward/metal_qwen35.rs");
        let source = std::fs::read_to_string(&path)
            .map_err(|reason| format!("{}: {reason}", path.display()))?;
        for (helper, methods) in helpers {
            validate_library_source(&source, helper, methods)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(target: &str, source: &str) -> Result<(), String> {
        validate_source(
            target,
            &StructuredSource::parse("indirect fixture", source, false)?,
        )
    }

    fn reject_fixture(target: &str, source: &str) {
        syn::parse_file(source).expect("mutation must remain syntactically valid Rust");
        assert!(fixture(target, source).is_err(), "accepted {source}");
    }

    const LOCAL: &str = r#"
fn run() {
    use lattice_inference::forward::metal_qwen35::bench_support;
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let mut state = State::new();
    let run_arm = |state: &mut MetalQwen35State, head: bool| -> Result<f64, String> {
        state.reset_state();
        let logits = bench_support::run_multimodal_prefill(state, request, cfg, head)?;
        Ok(logits.len() as f64)
    };
    run_arm(&mut state, true)?;
    run_arm(&mut state, false)?;
    for i in 0..4 { run_arm(&mut state, true)?; run_arm(&mut state, false)?; }
}
"#;

    #[test]
    fn indirect_local_closure_accepts_all_synchronous_invocations() {
        fixture(TARGETS[3], LOCAL).unwrap();
    }

    #[test]
    fn indirect_gdn_closures_require_both_complete_invocation_sets() {
        let source = r#"
fn run() {
    use lattice_inference::forward::metal_qwen35::bench_support::{self, GdnIsolatedChunkTiming};
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let mut state = State::new();
    let run_once = |state: &mut MetalQwen35State, tokens: &[u32]| {
        bench_support::forward_prefill_gdn_isolated_chunk(state, tokens, 0)
    };
    let run_once_production = |state: &mut MetalQwen35State, tokens: &[u32]| {
        bench_support::forward_prefill_production_chunk(state, tokens, 0)
    };
    run_once(&mut state, tokens); run_once(&mut state, tokens);
    run_once(&mut state, tokens); run_once(&mut state, tokens);
    run_once_production(&mut state, tokens); run_once_production(&mut state, tokens);
    run_once_production(&mut state, tokens); run_once_production(&mut state, tokens);
}
"#;
        fixture(TARGETS[2], source).unwrap();
        reject_fixture(
            TARGETS[2],
            &source.replacen("run_once_production(&mut state, tokens);", "", 1),
        );
        reject_fixture(
            TARGETS[2],
            &source.replace(
                "bench_support::forward_prefill_gdn_isolated_chunk",
                "other::forward_prefill_gdn_isolated_chunk",
            ),
        );
    }

    #[test]
    fn indirect_local_closures_pin_library_import_and_namespace() {
        fixture(TARGETS[3], LOCAL).unwrap();
        for mutant in [
            LOCAL.replace(
                "use lattice_inference::forward::metal_qwen35::bench_support;",
                "use other::bench_support;",
            ),
            LOCAL.replace(
                "use lattice_inference::forward::metal_qwen35::bench_support;",
                "use lattice_inference::forward::metal_qwen35::bench_support as replacement;",
            ),
            LOCAL.replace("let mut state", "mod lattice_inference {} let mut state"),
            LOCAL.replace(
                "let mut state",
                "use other::lattice_inference; let mut state",
            ),
            LOCAL.replace("let mut state", "use other::*; let mut state"),
            LOCAL.replace(
                "let mut state",
                "use other as lattice_inference; let mut state",
            ),
            LOCAL.replace(
                "use lattice_inference",
                "#[cfg(any())] use lattice_inference",
            ),
        ] {
            reject_fixture(TARGETS[3], &mutant);
        }
    }

    #[test]
    fn indirect_local_closure_rejects_escape_count_receiver_and_guard_mutants() {
        for mutant in [
            LOCAL.replace("let run_arm =", "let alias = run_arm; let run_arm ="),
            LOCAL.replace("run_arm(&mut state, true)?;", ""),
            LOCAL.replace("run_arm(&mut state, true)?;", "run_arm(&mut other, true)?;"),
            LOCAL.replace("let logits =", "let escaped = state; let logits ="),
            LOCAL.replace("let logits =", "let state = other; let logits ="),
            LOCAL.replace(
                "let logits = bench_support",
                "let logits = || bench_support",
            ),
            LOCAL.replace("let run_arm = |", "let run_arm = async |"),
            LOCAL.replace(
                "run_arm(&mut state, true)?;",
                "(|| run_arm(&mut state, true))()?;",
            ),
            LOCAL.replace("let mut state", "drop(_gpu); let mut state"),
            LOCAL.replace("let _gpu =", "#[cfg(feature = \"guard-only\")] let _gpu ="),
            LOCAL.replace(
                "let _gpu = lattice_inference::measurement::gpu_test_lock();",
                "{ let _gpu = lattice_inference::measurement::gpu_test_lock(); }",
            ),
        ] {
            reject_fixture(TARGETS[3], &mutant);
        }
    }

    const ITERATOR: &str = r#"
fn main() {
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let model = Model::new();
    let test_texts = ["one"];
    let pairs = [("a", "b", "c")];
    let all_embeddings: Vec<_> = test_texts.iter().map(|text| model.encode_all_layers(text)).collect();
    let pair_embeddings: Vec<_> = pairs.iter().map(|(a,b,c)| (model.encode_all_layers(a), model.encode_all_layers(b), model.encode_all_layers(c))).collect();
    model.encode_all_layers("timed");
}
"#;

    #[test]
    fn indirect_iterator_requires_eager_standard_collection() {
        fixture(TARGETS[0], ITERATOR).unwrap();
        for mutant in [
            ITERATOR.replace(".collect()", ""),
            ITERATOR.replace("Vec<_>", "Lazy<_>"),
            ITERATOR.replace(".map(|text|", ".map(async |text|"),
            ITERATOR.replace("model.encode_all_layers(a)", "other.encode_all_layers(a)"),
            ITERATOR.replace("let all_embeddings", "drop(_gpu); let all_embeddings"),
            ITERATOR.replace(
                "let test_texts = [\"one\"]",
                "let test_texts = custom_iterator()",
            ),
            ITERATOR.replace("fn main()", "struct Vec<T>(T); fn main()"),
        ] {
            reject_fixture(TARGETS[0], &mutant);
        }
    }

    const CALLBACK: &str = r#"
type Forward<'a> = dyn FnMut(&[EmbeddingGemma2Part], &[usize]) -> Result<(EmbeddingGemma2PartsEncoding, Option<VisionTimings>), Box<dyn std::error::Error>> + 'a;
fn main() { let model = Model::new(); dump(args, model, &mut |parts, widths| model.encode_parts_at_widths(parts, widths)); }
fn dump_metal() {
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let model = Model::new(); let mut state = State::new();
    dump(args, model, &mut |parts, widths| model.encode_parts_at_widths_metal(&mut state, parts, widths))
}
fn dump(args: &Args, model: &EmbeddingGemma2Model, forward: &mut Forward<'_>) -> Result<(), Box<dyn std::error::Error>> {
    for row in rows { let result = forward(parts, widths)?; }
    Ok(())
}
"#;

    #[test]
    fn indirect_borrowed_callback_accepts_synchronous_tail_call() {
        fixture(TARGETS[1], CALLBACK).unwrap();
    }

    #[test]
    fn indirect_callback_selects_the_active_cfg_disjoint_function() {
        let source = CALLBACK.replace(
            "fn dump_metal()",
            "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\nfn dump_metal()",
        );
        let source = format!(
            "{source}\n#[cfg(not(all(target_os = \"macos\", feature = \"metal-gpu\")))]\nfn dump_metal() {{ unavailable(); }}"
        );
        fixture(TARGETS[1], &source).unwrap();
        reject_fixture(
            TARGETS[1],
            &source.replace(
                "#[cfg(not(all(target_os = \"macos\", feature = \"metal-gpu\")))]",
                "",
            ),
        );
        reject_fixture(
            TARGETS[1],
            &CALLBACK.replace("fn dump_metal()", "#[cfg(any())]\nfn dump_metal()"),
        );
    }

    #[test]
    fn indirect_borrowed_callback_pins_owned_result_and_borrow_signature() {
        fixture(TARGETS[1], CALLBACK).unwrap();
        for mutant in [
            CALLBACK.replace(
                "Result<(EmbeddingGemma2PartsEncoding, Option<VisionTimings>)",
                "Result<Future<(EmbeddingGemma2PartsEncoding, Option<VisionTimings>)>",
            ),
            CALLBACK.replace("&[EmbeddingGemma2Part]", "&[OtherPart]"),
            CALLBACK.replace("dyn FnMut", "dyn FnOnce"),
            CALLBACK.replace("Forward<'_>", "Forward<'static>"),
            CALLBACK.replace(
                "-> Result<(), Box<dyn std::error::Error>>",
                "-> impl Future<Output = Result<(), Box<dyn std::error::Error>>>",
            ),
            CALLBACK.replace("type Forward", "#[cfg(any())] type Forward"),
        ] {
            reject_fixture(TARGETS[1], &mutant);
        }
    }

    #[test]
    fn indirect_borrowed_callback_rejects_escape_and_deferred_invocation() {
        for mutant in [
            CALLBACK.replace("let result = forward", "let alias = forward; let result = alias"),
            CALLBACK.replace("let result = forward(parts, widths)?;", "let result = || forward(parts, widths);"),
            CALLBACK.replace("let result = forward(parts, widths)?;", "forward(parts, widths)?; forward(parts, widths)?;"),
            CALLBACK.replace("fn dump(args", "async fn dump(args"),
            CALLBACK.replace("&mut |parts, widths| model.encode_parts_at_widths_metal(&mut state, parts, widths)", "callback"),
            CALLBACK.replace("&mut Forward", "Forward"),
            CALLBACK.replace("dyn FnMut", "dyn Future"),
            format!("{CALLBACK}\nfn escape() {{ dump(args, model, callback); }}"),
        ] {
            reject_fixture(TARGETS[1], &mutant);
        }
    }

    const HELPER: &str = r#"
use lattice_inference::model::qwen35::{PerplexityConfig, PerplexityReport, Qwen35Model};
fn main() {
    let model = match Qwen35Model::from_safetensors(&model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: failed to load model: {e}");
            return ExitCode::FAILURE;
        }
    };
    model.compute_perplexity(tokens, cfg);
}
fn run_metal_q4() {
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let mut state = State::new(); run_on_metal_state(&mut state)
}
fn run_metal_model_dir() {
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let mut state = State::new(); run_on_metal_state(&mut state)
}
fn run_on_metal_state(state: &mut MetalQwen35State) -> Result<Report, Error> {
    state.load_lora_adapter(layers, scale, seed)?;
    state.compute_perplexity(tokens, cfg)
}
"#;

    #[test]
    fn indirect_same_file_helper_requires_complete_guarded_call_graph() {
        fixture(TARGETS[4], HELPER).unwrap();
        for mutant in [
            format!("{HELPER}\nfn extra() {{ run_on_metal_state(&mut state); }}"),
            HELPER.replace(
                "    state.compute_perplexity(tokens, cfg)\n",
                "    let deferred = || state.compute_perplexity(tokens, cfg); Ok(report)\n",
            ),
            HELPER.replace("let mut state", "drop(_gpu); let mut state"),
            HELPER.replace("state.load_lora_adapter", "other.load_lora_adapter"),
            HELPER.replace("fn run_on_metal_state", "async fn run_on_metal_state"),
            HELPER.replace(
                "state.load_lora_adapter",
                "let alias = state; state.load_lora_adapter",
            ),
        ] {
            reject_fixture(TARGETS[4], &mutant);
        }
    }

    #[test]
    fn indirect_perplexity_cpu_exception_requires_receiver_provenance() {
        fixture(TARGETS[4], HELPER).unwrap();
        let replacement = HELPER.replace("Qwen35Model::from_safetensors(&model_dir)", "{ let _setup = lattice_inference::measurement::gpu_test_lock(); MetalQwen35State::from_q4_dir(dir, tokenizer, cfg, context) }");
        assert_ne!(replacement, HELPER);
        assert_eq!(replacement.matches("model.compute_perplexity(").count(), 1);
        reject_fixture(TARGETS[4], &replacement);
        reject_fixture(
            TARGETS[4],
            &format!("use crate::metal_backend as lattice_inference;\n{HELPER}"),
        );
        reject_fixture(
            TARGETS[4],
            &format!("type Qwen35Model = MetalQwen35State;\n{HELPER}"),
        );
        reject_fixture(
            TARGETS[4],
            &HELPER.replace("let model = match", "let model = other; let model = match"),
        );
    }

    #[test]
    fn indirect_trusted_library_helper_rejects_async_and_state_escape() {
        let source = r#"mod inner { pub mod bench_support {
        pub fn run_multimodal_prefill(state: &mut MetalQwen35State, request: &crate::vision::multimodal::Qwen35VisionRequest, cfg: &Qwen35Config, always_emit_head: bool) -> Result<Vec<f32>, String> {
            let a = state.forward_step_injected_mrope();
            let b = state.forward_step_mrope(); Ok(b)
        } } } pub use inner::bench_support;"#;
        let methods = &["forward_step_injected_mrope", "forward_step_mrope"];
        validate_library_source(source, "run_multimodal_prefill", methods).unwrap();
        for mutant in [
            source.replace(
                "fn run_multimodal_prefill",
                "async fn run_multimodal_prefill",
            ),
            source.replace("let a =", "let escaped = state; let a ="),
            source.replace("let a =", "escape!(state); let a ="),
            source.replace("let a = state", "let a = || state"),
            source.replace("let b = state.forward_step_mrope();", ""),
            source.replace("mod inner", "mod moved"),
            source.replace("pub mod bench_support", "pub mod renamed"),
            source.replace(
                "pub use inner::bench_support;",
                "pub use other::bench_support;",
            ),
            source.replace(
                "pub fn run_multimodal_prefill",
                "#[cfg(feature = \"unreviewed\")] pub fn run_multimodal_prefill",
            ),
            source.replace(
                "Result<Vec<f32>, String>",
                "Result<Future<Vec<f32>>, String>",
            ),
        ] {
            syn::parse_file(&mutant).expect("library mutation must remain syntactically valid");
            assert!(
                validate_library_source(&mutant, "run_multimodal_prefill", methods).is_err(),
                "accepted {mutant}"
            );
        }
    }

    #[test]
    fn indirect_dispatch_rejects_unknown_missing_and_moved_work() {
        assert!(validate_target("examples/unknown.rs", &[]).is_err());
        assert!(validate_target(TARGETS[3], &[]).is_err());
        let sources = [
            ParsedModuleSource {
                relative: TARGETS[3].into(),
                parsed: StructuredSource::parse("root", LOCAL, false).unwrap(),
            },
            ParsedModuleSource {
                relative: "moved.rs".into(),
                parsed: StructuredSource::parse(
                    "moved",
                    "fn hidden() { state.compute_perplexity(); }",
                    false,
                )
                .unwrap(),
            },
        ];
        assert!(validate_target(TARGETS[3], &sources).is_err());
    }
}
