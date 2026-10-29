//! Source contract for the callback-mediated Metal text dump.

use syn::visit::{self, Visit};
use syn::{Expr, FnArg, Item, Meta, Pat, Stmt, Type};

const SOURCE: &str = include_str!("../examples/dump_embeddinggemma2.rs");

fn path_is(expr: &Expr, names: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == names.len()
        && path.path.segments.iter().zip(names).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
        })
}

fn metal_cfg(attrs: &[syn::Attribute]) -> bool {
    let [attr] = attrs else { return false };
    if !attr.path().is_ident("cfg") {
        return false;
    }
    let Ok(Meta::List(all)) = attr.parse_args::<Meta>() else {
        return false;
    };
    if !all.path.is_ident("all") {
        return false;
    }
    let Ok(args) =
        all.parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
    else {
        return false;
    };
    let pair = |meta: &Meta, key: &str, value: &str| {
        matches!(meta,
        Meta::NameValue(nv) if nv.path.is_ident(key) && matches!(&nv.value,
            Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == value)))
    };
    args.len() == 2
        && pair(&args[0], "target_os", "macos")
        && pair(&args[1], "feature", "metal-gpu")
}

fn guard(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Call(call) = init.expr.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "_gpu"
        && binding.by_ref.is_none()
        && binding.mutability.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && call.attrs.is_empty()
        && call.args.is_empty()
        && path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        )
}

fn state_binding(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Try(try_expr) = init.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = try_expr.expr.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && try_expr.attrs.is_empty()
        && call.attrs.is_empty()
        && call.args.len() == 1
        && call
            .args
            .first()
            .is_some_and(|arg| path_is(arg, &["model"]))
        && path_is(&call.func, &["MetalEmbeddingGemma2State", "new"])
}

fn sensitive(tokens: proc_macro2::TokenStream, names: &[&str]) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => names.iter().any(|name| id == *name),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream(), names),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let captured: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    names.contains(&captured.as_str())
                })
            })
        }
        _ => false,
    })
}

fn direct_callback(stmt: &Stmt) -> Option<&syn::ExprClosure> {
    let Stmt::Expr(Expr::Call(call), None) = stmt else {
        return None;
    };
    if !call.attrs.is_empty()
        || !path_is(&call.func, &["dump"])
        || call.args.len() != 3
        || !path_is(&call.args[0], &["args"])
        || !path_is(&call.args[1], &["model"])
    {
        return None;
    }
    let Expr::Reference(reference) = &call.args[2] else {
        return None;
    };
    let Expr::Closure(closure) = reference.expr.as_ref() else {
        return None;
    };
    if !reference.attrs.is_empty()
        || reference.mutability.is_none()
        || !closure.attrs.is_empty()
        || closure.asyncness.is_some()
        || closure.capture.is_some()
        || closure.inputs.len() != 3
    {
        return None;
    }
    for (pat, expected) in closure.inputs.iter().zip(["ids", "widths", "want_hidden"]) {
        let Pat::Ident(binding) = pat else {
            return None;
        };
        if binding.ident != expected
            || binding.mutability.is_some()
            || binding.by_ref.is_some()
            || binding.subpat.is_some()
        {
            return None;
        }
    }
    Some(closure)
}

// `bool::then` calls this closure immediately. Pin its exact expression and
// callback type so a stored or deferred replacement cannot be mistaken for it.
fn hidden_then(closure: &syn::ExprClosure) -> bool {
    let Expr::Block(body) = closure.body.as_ref() else {
        return false;
    };
    let Some(Stmt::Local(local)) = body.block.stmts.get(1) else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Try(try_expr) = init.expr.as_ref() else {
        return false;
    };
    let Expr::MethodCall(transpose) = try_expr.expr.as_ref() else {
        return false;
    };
    let Expr::MethodCall(then) = transpose.receiver.as_ref() else {
        return false;
    };
    let Some(Expr::Closure(hidden)) = then.args.first() else {
        return false;
    };
    let Expr::MethodCall(work) = hidden.body.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "hidden"
        && binding.by_ref.is_none()
        && binding.mutability.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && try_expr.attrs.is_empty()
        && transpose.attrs.is_empty()
        && transpose.method == "transpose"
        && transpose.args.is_empty()
        && then.attrs.is_empty()
        && then.method == "then"
        && path_is(&then.receiver, &["want_hidden"])
        && then.args.len() == 1
        && hidden.attrs.is_empty()
        && hidden.inputs.is_empty()
        && hidden.asyncness.is_none()
        && hidden.capture.is_none()
        && work.attrs.is_empty()
        && work.method == "token_states_metal"
        && path_is(&work.receiver, &["model"])
        && work.args.len() == 2
        && matches!(&work.args[0], Expr::Reference(r) if r.attrs.is_empty()
            && r.mutability.is_some() && path_is(&r.expr, &["state"]))
        && path_is(&work.args[1], &["ids"])
}

fn forward_accepts_bool(file: &syn::File) -> bool {
    let aliases = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Type(alias) if alias.ident == "Forward" => Some(alias),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [alias] = aliases.as_slice() else {
        return false;
    };
    let Type::TraitObject(object) = alias.ty.as_ref() else {
        return false;
    };
    let mut signatures = object.bounds.iter().filter_map(|bound| match bound {
        syn::TypeParamBound::Trait(trait_bound)
            if trait_bound.path.segments.len() == 1
                && trait_bound.path.segments[0].ident == "FnMut" =>
        {
            match &trait_bound.path.segments[0].arguments {
                syn::PathArguments::Parenthesized(args) => Some(args),
                _ => None,
            }
        }
        _ => None,
    });
    let Some(signature) = signatures.next() else {
        return false;
    };
    if signatures.next().is_some() || signature.inputs.len() != 3 {
        return false;
    }
    matches!(&signature.inputs[2], Type::Path(path) if path.qself.is_none() && path.path.is_ident("bool"))
}

#[derive(Default)]
struct MetalWork {
    guards: usize,
    guard_calls: usize,
    state_bindings: usize,
    state_paths: usize,
    model_paths: usize,
    hidden_bindings: usize,
    hidden_paths: usize,
    constructors: usize,
    dump_calls: usize,
    encode: usize,
    hidden: usize,
    attrs: usize,
    depth: usize,
    deferred: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for MetalWork {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attrs += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.guards += usize::from(pat.ident == "_gpu");
        self.state_bindings += usize::from(pat.ident == "state");
        self.hidden_bindings += usize::from(pat.ident == "want_hidden");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.state_paths += usize::from(path.path.is_ident("state"));
        self.model_paths += usize::from(path.path.is_ident("model"));
        self.hidden_paths += usize::from(path.path.is_ident("want_hidden"));
        self.bad |= path.path.is_ident("_gpu");
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.constructors +=
            usize::from(path_is(&call.func, &["MetalEmbeddingGemma2State", "new"]));
        self.dump_calls += usize::from(path_is(&call.func, &["dump"]));
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "encode_ids_at_widths_metal" || call.method == "token_states_metal" {
            let is_encode = call.method == "encode_ids_at_widths_metal";
            let state_arg = matches!(call.args.first(), Some(Expr::Reference(r)) if r.attrs.is_empty()
                && r.mutability.is_some() && path_is(&r.expr, &["state"]));
            self.bad |= !path_is(&call.receiver, &["model"])
                || !call.attrs.is_empty()
                || self.deferred != 0
                || !state_arg
                || self.depth != if is_encode { 1 } else { 2 }
                || call.args.len() != if is_encode { 3 } else { 2 };
            self.encode += usize::from(is_encode);
            self.hidden += usize::from(!is_encode);
        } else if path_is(&call.receiver, &["state"]) || path_is(&call.receiver, &["model"]) {
            self.bad = true;
        }
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        self.depth += 1;
        visit::visit_expr_closure(self, expr);
        self.depth -= 1;
    }
    fn visit_expr_async(&mut self, expr: &'ast syn::ExprAsync) {
        self.deferred += 1;
        visit::visit_expr_async(self, expr);
        self.deferred -= 1;
    }
    fn visit_item(&mut self, item: &'ast syn::Item) {
        self.deferred += 1;
        visit::visit_item(self, item);
        self.deferred -= 1;
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(
            mac.tokens.clone(),
            &[
                "_gpu",
                "state",
                "model",
                "dump",
                "want_hidden",
                "encode_ids_at_widths_metal",
                "token_states_metal",
            ],
        );
    }
}

#[derive(Default)]
struct ForwardUse {
    paths: usize,
    calls: usize,
    bindings: usize,
    attrs: usize,
    deferred: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for ForwardUse {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attrs += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.bindings += usize::from(pat.ident == "forward");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.paths += usize::from(path.path.is_ident("forward"));
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(&call.func, &["forward"]) {
            self.calls += 1;
            self.bad |= self.deferred > 0 || !call.attrs.is_empty() || call.args.len() != 3;
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        self.deferred += 1;
        visit::visit_expr_closure(self, expr);
        self.deferred -= 1;
    }
    fn visit_expr_async(&mut self, expr: &'ast syn::ExprAsync) {
        self.deferred += 1;
        visit::visit_expr_async(self, expr);
        self.deferred -= 1;
    }
    fn visit_item(&mut self, item: &'ast syn::Item) {
        self.deferred += 1;
        visit::visit_item(self, item);
        self.deferred -= 1;
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(mac.tokens.clone(), &["forward"]);
    }
}

#[derive(Default)]
struct AllWork {
    encode: usize,
    hidden: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for AllWork {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.encode += usize::from(call.method == "encode_ids_at_widths_metal");
        self.hidden += usize::from(call.method == "token_states_metal");
        visit::visit_expr_method_call(self, call);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(
            mac.tokens.clone(),
            &["encode_ids_at_widths_metal", "token_states_metal"],
        );
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let find = |name: &str| {
        file.items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(f) if f.sig.ident == name => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let metal = find("dump_metal")
        .into_iter()
        .filter(|f| metal_cfg(&f.attrs))
        .collect::<Vec<_>>();
    let dumps = find("dump");
    let ([metal], [dump]) = (metal.as_slice(), dumps.as_slice()) else {
        return Err("functions changed");
    };
    if metal.sig.asyncness.is_some()
        || dump.sig.asyncness.is_some()
        || !dump.attrs.is_empty()
        || !forward_accepts_bool(&file)
    {
        return Err("callback type or execution changed");
    }
    let Some(FnArg::Typed(arg)) = dump.sig.inputs.last() else {
        return Err("callback parameter changed");
    };
    let Pat::Ident(pat) = arg.pat.as_ref() else {
        return Err("callback parameter changed");
    };
    let Type::Reference(reference) = arg.ty.as_ref() else {
        return Err("callback parameter changed");
    };
    let Type::Path(path) = reference.elem.as_ref() else {
        return Err("callback parameter changed");
    };
    if pat.ident != "forward"
        || pat.by_ref.is_some()
        || pat.subpat.is_some()
        || reference.mutability.is_none()
        || path.qself.is_some()
        || path.path.segments.len() != 1
        || path.path.segments[0].ident != "Forward"
    {
        return Err("callback parameter changed");
    }
    let guards = metal
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| guard(stmt))
        .collect::<Vec<_>>();
    let states = metal
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| state_binding(stmt))
        .collect::<Vec<_>>();
    let ([(g, _)], [(s, _)]) = (guards.as_slice(), states.as_slice()) else {
        return Err("guard or state changed");
    };
    let Some(callback) = metal.block.stmts.last().and_then(direct_callback) else {
        return Err("callback changed");
    };
    if g >= s || !hidden_then(callback) {
        return Err("guard or immediate hidden callback changed");
    }
    let mut work = MetalWork::default();
    work.visit_block(&metal.block);
    let mut helper = ForwardUse::default();
    helper.visit_block(&dump.block);
    let mut global = AllWork::default();
    global.visit_file(&file);
    if work.bad
        || work.attrs != 0
        || work.guards != 1
        || work.guard_calls != 1
        || work.state_bindings != 1
        || work.state_paths != 2
        || work.model_paths != 4
        || work.hidden_bindings != 1
        || work.hidden_paths != 1
        || work.constructors != 1
        || work.dump_calls != 1
        || work.encode != 1
        || work.hidden != 1
        || helper.bad
        || helper.attrs != 0
        || helper.bindings != 0
        || helper.paths != 1
        || helper.calls != 1
        || global.bad
        || global.encode != 1
        || global.hidden != 1
    {
        return Err("unreviewed callback, guard or work use");
    }
    Ok(())
}

#[test]
fn text_dump_callbacks_keep_metal_work_under_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn text_dump_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu = lattice_inference::measurement::gpu_test_lock();\n";
    let helper = "        let (embeddings, hidden) = forward(&ids, &args.widths, want_hidden)?;";
    let encode =
        "        let embeddings = model.encode_ids_at_widths_metal(&mut state, ids, widths)?;";
    let hidden = "            .then(|| model.token_states_metal(&mut state, ids))";
    let callback = "dump(args, model, &mut |ids, widths, want_hidden|";
    let bool_type = "        bool,\n    ) -> Result";
    assert!(
        SOURCE.contains(guard)
            && SOURCE.contains(helper)
            && SOURCE.contains(encode)
            && SOURCE.contains(hidden)
            && SOURCE.contains(callback)
            && SOURCE.contains(bool_type)
    );
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu = ();\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu);\n"), 1),
        SOURCE.replacen(
            encode,
            "        let embeddings = other.encode_ids_at_widths_metal(&mut state, ids, widths)?;",
            1,
        ),
        SOURCE.replacen(encode, &format!("{encode}\n{encode}"), 1),
        SOURCE.replacen(
            hidden,
            "            .then(|| model.token_states_metal(&mut other, ids))",
            1,
        ),
        SOURCE.replacen(
            hidden,
            "            .then(|| other.token_states_metal(&mut state, ids))",
            1,
        ),
        SOURCE.replacen(
            hidden,
            "            .then(move || model.token_states_metal(&mut state, ids))",
            1,
        ),
        SOURCE.replacen(
            hidden,
            "            .then(|| model.token_states_metal(&mut state, ids)).map(|x| x)",
            1,
        ),
        SOURCE.replacen(bool_type, "        usize,\n    ) -> Result", 1),
        SOURCE.replacen(helper, "", 1),
        SOURCE.replacen(helper, &format!("{helper}\n{helper}"), 1),
        SOURCE.replacen(
            helper,
            "        let deferred = || forward(&ids, &args.widths, want_hidden);",
            1,
        ),
        SOURCE.replacen(helper, &format!("        #[cfg(any())]\n{helper}"), 1),
        SOURCE.replacen(helper, &format!("{helper}\n        consume!(forward);"), 1),
        SOURCE.replacen(guard, "", 1).replacen(
            "    let mut state = MetalEmbeddingGemma2State::new(model)?;",
            &format!("    let mut state = MetalEmbeddingGemma2State::new(model)?;\n{guard}"),
            1,
        ),
        SOURCE.replacen(
            callback,
            "dump(args, model, &mut move |ids, widths, want_hidden|",
            1,
        ),
    ] {
        assert!(check(&mutated).is_err());
    }
}
