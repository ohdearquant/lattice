//! Source contract for the callback-mediated Metal vision dump.

use syn::Meta;
use syn::visit::{self, Visit};
use syn::{Expr, FnArg, Item, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/dump_embeddinggemma2_vision.rs");

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

fn callback(stmt: &Stmt) -> Option<&syn::ExprClosure> {
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
        || closure.inputs.len() != 2
    {
        return None;
    }
    let (Some(Pat::Ident(parts)), Some(Pat::Ident(widths))) =
        (closure.inputs.first(), closure.inputs.last())
    else {
        return None;
    };
    if parts.ident != "parts"
        || widths.ident != "widths"
        || parts.mutability.is_some()
        || widths.mutability.is_some()
        || parts.by_ref.is_some()
        || widths.by_ref.is_some()
        || parts.subpat.is_some()
        || widths.subpat.is_some()
    {
        return None;
    }
    Some(closure)
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

#[derive(Default)]
struct MetalWork {
    guards: usize,
    guard_calls: usize,
    state_bindings: usize,
    state_paths: usize,
    model_paths: usize,
    constructors: usize,
    dump_calls: usize,
    work: usize,
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
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.state_paths += usize::from(path.path.is_ident("state"));
        self.model_paths += usize::from(path.path.is_ident("model"));
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
        if call.method == "encode_parts_at_widths_metal" {
            let state_arg = matches!(call.args.first(), Some(Expr::Reference(r))
                if r.attrs.is_empty() && r.mutability.is_some() && path_is(&r.expr, &["state"]));
            self.bad |= !path_is(&call.receiver, &["model"])
                || !call.attrs.is_empty()
                || self.depth != 1
                || self.deferred != 0
                || call.args.len() != 4
                || !state_arg;
            self.work += 1;
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
                "encode_parts_at_widths_metal",
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
            self.bad |= self.deferred > 0 || !call.attrs.is_empty() || call.args.len() != 2;
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
    calls: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for AllWork {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.calls += usize::from(call.method == "encode_parts_at_widths_metal");
        visit::visit_expr_method_call(self, call);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(mac.tokens.clone(), &["encode_parts_at_widths_metal"]);
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
    if metal.sig.asyncness.is_some() || dump.sig.asyncness.is_some() || !dump.attrs.is_empty() {
        return Err("deferred or conditional function");
    }
    let last = dump.sig.inputs.last();
    if !matches!(last, Some(FnArg::Typed(arg)) if matches!(arg.pat.as_ref(), Pat::Ident(pat)
        if pat.ident == "forward" && pat.by_ref.is_none() && pat.subpat.is_none()))
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
    if g >= s || metal.block.stmts.last().and_then(callback).is_none() {
        return Err("guard or direct callback changed");
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
        || work.state_paths != 1
        || work.model_paths != 3
        || work.constructors != 1
        || work.dump_calls != 1
        || work.work != 1
        || helper.bad
        || helper.attrs != 0
        || helper.bindings != 0
        || helper.paths != 1
        || helper.calls != 1
        || global.bad
        || global.calls != 1
    {
        return Err("unreviewed callback, guard or work use");
    }
    Ok(())
}

#[test]
fn vision_dump_callback_keeps_metal_work_under_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn vision_dump_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu = lattice_inference::measurement::gpu_test_lock();\n";
    let call = "        let (encoding, timings) = forward(&parts, &args.widths)?;";
    let work = "            model.encode_parts_at_widths_metal(&mut state, parts, widths, &mut || false)?;";
    assert!(SOURCE.contains(guard) && SOURCE.contains(call) && SOURCE.contains(work));
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu = ();\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu);\n"), 1),
        SOURCE.replacen(work, "            other.encode_parts_at_widths_metal(&mut state, parts, widths, &mut || false)?;", 1),
        SOURCE.replacen(work, "            model.encode_parts_at_widths_metal(&mut other, parts, widths, &mut || false)?;", 1),
        SOURCE.replacen(work, &format!("{work}\n{work}"), 1),
        SOURCE.replacen(call, "", 1),
        SOURCE.replacen(call, &format!("{call}\n{call}"), 1),
        SOURCE.replacen(call, "        let deferred = || forward(&parts, &args.widths);", 1),
        SOURCE.replacen(call, &format!("        #[cfg(any())]\n{call}"), 1),
        SOURCE.replacen(call, &format!("{call}\n        consume!(forward);"), 1),
        SOURCE.replacen(guard, "", 1).replacen("    let mut state = MetalEmbeddingGemma2State::new(model)?;", &format!("    let mut state = MetalEmbeddingGemma2State::new(model)?;\n{guard}"), 1),
        SOURCE.replacen("dump(args, model, &mut |parts, widths|", "dump(args, model, &mut move |parts, widths|", 1),
    ] { assert!(check(&mutated).is_err()); }
}
