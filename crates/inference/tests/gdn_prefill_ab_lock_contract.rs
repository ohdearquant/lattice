//! Source contract for the GDN prefill benchmark work closures.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt, Type};

const SOURCE: &str = include_str!("../src/bin/bench_gdn_prefill_ab.rs");

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
    args.len() == 3
        && pair(&args[0], "target_os", "macos")
        && pair(&args[1], "feature", "metal-gpu")
        && pair(&args[2], "feature", "bench-internals")
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
        && binding.ident == "_gpu_lock"
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
    let Expr::MethodCall(map_err) = try_expr.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = map_err.receiver.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && try_expr.attrs.is_empty()
        && map_err.attrs.is_empty()
        && map_err.method == "map_err"
        && map_err.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == 3
        && path_is(&call.func, &["MetalQwen35State", "new"])
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "_gpu_lock"
                | "gpu_test_lock"
                | "state"
                | "run_once"
                | "run_once_production"
                | "reset_state"
                | "set_gdn_chunked"
                | "forward_prefill_gdn_isolated_chunk"
                | "forward_prefill_production_chunk"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let captured: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    matches!(
                        captured.as_str(),
                        "_gpu_lock" | "state" | "run_once" | "run_once_production"
                    )
                })
            })
        }
        _ => false,
    })
}

fn arm_binding<'a>(stmt: &'a Stmt, name: &str) -> Option<&'a syn::ExprClosure> {
    let Stmt::Local(local) = stmt else {
        return None;
    };
    let Pat::Ident(binding) = &local.pat else {
        return None;
    };
    let Some(init) = &local.init else { return None };
    let Expr::Closure(closure) = init.expr.as_ref() else {
        return None;
    };
    if !local.attrs.is_empty()
        || binding.ident != name
        || binding.by_ref.is_some()
        || binding.mutability.is_some()
        || binding.subpat.is_some()
        || init.diverge.is_some()
        || !closure.attrs.is_empty()
        || closure.asyncness.is_some()
        || closure.capture.is_some()
        || closure.inputs.len() != 2
    {
        return None;
    }
    let (Some(Pat::Type(state)), Some(Pat::Type(tokens))) =
        (closure.inputs.first(), closure.inputs.last())
    else {
        return None;
    };
    let (Pat::Ident(state_name), Pat::Ident(token_name)) =
        (state.pat.as_ref(), tokens.pat.as_ref())
    else {
        return None;
    };
    let (Type::Reference(state_ref), Type::Reference(token_ref)) =
        (state.ty.as_ref(), tokens.ty.as_ref())
    else {
        return None;
    };
    let Type::Path(state_ty) = state_ref.elem.as_ref() else {
        return None;
    };
    let Type::Slice(token_slice) = token_ref.elem.as_ref() else {
        return None;
    };
    let Type::Path(token_ty) = token_slice.elem.as_ref() else {
        return None;
    };
    if state_name.ident != "state"
        || token_name.ident != "tokens"
        || state_name.by_ref.is_some()
        || token_name.by_ref.is_some()
        || state_name.mutability.is_some()
        || token_name.mutability.is_some()
        || state_name.subpat.is_some()
        || token_name.subpat.is_some()
        || state_ref.mutability.is_none()
        || token_ref.mutability.is_some()
        || state_ty.qself.is_some()
        || !state_ty.path.is_ident("MetalQwen35State")
        || token_ty.qself.is_some()
        || !token_ty.path.is_ident("u32")
    {
        return None;
    }
    Some(closure)
}

fn allowed_attribute(attr: &syn::Attribute) -> bool {
    if attr.path().is_ident("cfg") {
        return attr
            .parse_args::<syn::Path>()
            .is_ok_and(|path| path.is_ident("debug_assertions"));
    }
    if attr.path().is_ident("derive") {
        return attr
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|paths| {
                paths.len() == 2 && paths[0].is_ident("Clone") && paths[1].is_ident("Copy")
            });
    }
    false
}

fn debug_compile_guard(stmt: &Stmt) -> bool {
    let Stmt::Macro(mac) = stmt else { return false };
    mac.mac.path.is_ident("compile_error")
        && mac.attrs.len() == 1
        && mac.attrs[0].path().is_ident("cfg")
        && allowed_attribute(&mac.attrs[0])
}

#[derive(Default)]
struct Work {
    guards: usize,
    guard_calls: usize,
    states: usize,
    state_paths: usize,
    arm_bindings: [usize; 2],
    arm_paths: [usize; 2],
    arm_calls: [usize; 2],
    constructors: usize,
    helpers: [usize; 2],
    queries: usize,
    reset: usize,
    set: usize,
    attrs: usize,
    depth: usize,
    deferred: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for Work {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attrs += 1;
        self.bad |= !allowed_attribute(attr);
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.guards += usize::from(pat.ident == "_gpu_lock");
        self.states += usize::from(pat.ident == "state");
        self.arm_bindings[0] += usize::from(pat.ident == "run_once");
        self.arm_bindings[1] += usize::from(pat.ident == "run_once_production");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.state_paths += usize::from(path.path.is_ident("state"));
        self.arm_paths[0] += usize::from(path.path.is_ident("run_once"));
        self.arm_paths[1] += usize::from(path.path.is_ident("run_once_production"));
        self.bad |= path.path.is_ident("_gpu_lock");
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.constructors += usize::from(path_is(&call.func, &["MetalQwen35State", "new"]));
        for (i, name) in ["run_once", "run_once_production"].iter().enumerate() {
            if path_is(&call.func, &[*name]) {
                self.arm_calls[i] += 1;
                self.bad |= self.depth != 0
                    || self.deferred != 0
                    || !call.attrs.is_empty()
                    || call.args.len() != 2;
                if call.args.len() == 2 {
                    self.bad |= !matches!(&call.args[0], Expr::Reference(r) if r.mutability.is_some() && path_is(&r.expr, &["state"]))
                        || !matches!(&call.args[1], Expr::Reference(r) if r.mutability.is_none() && path_is(&r.expr, &["tokens"]));
                }
            }
        }
        for (i, name) in [
            "forward_prefill_gdn_isolated_chunk",
            "forward_prefill_production_chunk",
        ]
        .iter()
        .enumerate()
        {
            if path_is(&call.func, &["bench_support", *name]) {
                self.helpers[i] += 1;
                self.bad |= self.depth != 1
                    || self.deferred != 0
                    || !call.attrs.is_empty()
                    || call.args.len() != 3;
                if call.args.len() == 3 {
                    self.bad |= !path_is(&call.args[0], &["state"])
                        || !path_is(&call.args[1], &["chunk"])
                        || !path_is(&call.args[2], &[if i == 0 { "start" } else { "pos" }]);
                }
            }
        }
        for name in ["gdn_chunked_prefill_supported", "max_prefill", "vocab_size"] {
            if path_is(&call.func, &["bench_support", name]) {
                self.queries += 1;
                self.bad |= self.depth != 0
                    || self.deferred != 0
                    || !call.attrs.is_empty()
                    || call.args.len() != 1
                    || !matches!(call.args.first(), Some(Expr::Reference(r))
                        if r.mutability.is_none() && path_is(&r.expr, &["state"]));
            }
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "reset_state" || call.method == "set_gdn_chunked" {
            self.bad |= !path_is(&call.receiver, &["state"])
                || self.depth != 0
                || self.deferred != 0
                || !call.attrs.is_empty();
            self.reset += usize::from(call.method == "reset_state");
            self.set += usize::from(call.method == "set_gdn_chunked");
        } else if path_is(&call.receiver, &["state"]) {
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
        self.bad |= sensitive(mac.tokens.clone());
    }
}

#[derive(Default)]
struct Global {
    helpers: [usize; 2],
    calls: [usize; 2],
    reset: usize,
    set: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for Global {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        for (i, name) in ["run_once", "run_once_production"].iter().enumerate() {
            self.calls[i] += usize::from(path_is(&call.func, &[*name]));
        }
        for (i, name) in [
            "forward_prefill_gdn_isolated_chunk",
            "forward_prefill_production_chunk",
        ]
        .iter()
        .enumerate()
        {
            self.helpers[i] += usize::from(path_is(&call.func, &["bench_support", *name]));
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.reset += usize::from(call.method == "reset_state");
        self.set += usize::from(call.method == "set_gdn_chunked");
        visit::visit_expr_method_call(self, call);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(mac.tokens.clone());
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let find = |name: &str| {
        file.items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(f) if f.sig.ident == name && metal_cfg(&f.attrs) => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let mains = find("main");
    let runs = find("run");
    let ([main], [run]) = (mains.as_slice(), runs.as_slice()) else {
        return Err("cfg entry changed");
    };
    if main.sig.asyncness.is_some() || run.sig.asyncness.is_some() {
        return Err("async entry");
    }
    let [Stmt::Expr(Expr::If(entry), None)] = main.block.stmts.as_slice() else {
        return Err("main entry changed");
    };
    let Expr::Let(condition) = entry.cond.as_ref() else {
        return Err("main condition changed");
    };
    let Expr::Call(call) = condition.expr.as_ref() else {
        return Err("main call changed");
    };
    if !entry.attrs.is_empty()
        || entry.else_branch.is_some()
        || !condition.attrs.is_empty()
        || !call.attrs.is_empty()
        || !path_is(&call.func, &["run"])
        || !call.args.is_empty()
    {
        return Err("main call changed");
    }
    if run
        .block
        .stmts
        .iter()
        .filter(|stmt| debug_compile_guard(stmt))
        .count()
        != 1
    {
        return Err("release-only compile guard changed");
    }
    let guards = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| guard(stmt))
        .collect::<Vec<_>>();
    let states = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| state_binding(stmt))
        .collect::<Vec<_>>();
    let isolated = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(i, stmt)| arm_binding(stmt, "run_once").map(|arm| (i, arm)))
        .collect::<Vec<_>>();
    let production = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(i, stmt)| arm_binding(stmt, "run_once_production").map(|arm| (i, arm)))
        .collect::<Vec<_>>();
    let ([(g, _)], [(s, _)], [(a, first)], [(b, second)]) = (
        guards.as_slice(),
        states.as_slice(),
        isolated.as_slice(),
        production.as_slice(),
    ) else {
        return Err("guard, state or closure changed");
    };
    if !(g < s && s < a && a < b) {
        return Err("guard and state must precede both closures");
    }
    let mut all = Work::default();
    all.visit_block(&run.block);
    let mut first_work = Work::default();
    first_work.visit_expr_closure(first);
    let mut second_work = Work::default();
    second_work.visit_expr_closure(second);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*b] {
        before.visit_stmt(stmt);
    }
    let mut global = Global::default();
    global.visit_file(&file);
    if all.bad
        || all.attrs != 2
        || all.guards != 1
        || all.guard_calls != 1
        || all.states != 3
        || all.state_paths != 27
        || all.arm_bindings != [1, 1]
        || all.arm_paths != [4, 4]
        || all.arm_calls != [4, 4]
        || all.constructors != 1
        || all.helpers != [1, 1]
        || all.queries != 3
        || all.reset != 8
        || all.set != 6
        || first_work.bad
        || first_work.states != 1
        || first_work.state_paths != 1
        || first_work.helpers != [1, 0]
        || first_work.arm_calls != [0, 0]
        || second_work.bad
        || second_work.states != 1
        || second_work.state_paths != 1
        || second_work.helpers != [0, 1]
        || second_work.arm_calls != [0, 0]
        || before.arm_calls != [0, 0]
        || global.bad
        || global.helpers != [1, 1]
        || global.calls != [4, 4]
        || global.reset != 8
        || global.set != 6
    {
        return Err("unreviewed guard, closure or work use");
    }
    Ok(())
}

#[test]
fn gdn_prefill_closures_and_calls_keep_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn gdn_prefill_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "                    state.reset_state();";
    let isolated = "            let t = bench_support::forward_prefill_gdn_isolated_chunk(state, chunk, start)?;";
    let production =
        "            bench_support::forward_prefill_production_chunk(state, chunk, pos)?;";
    let arm = "                    let _ = run_once_production(&mut state, &tokens)?;";
    let closure = "let run_once = |state: &mut MetalQwen35State,";
    assert!(
        SOURCE.contains(guard)
            && SOURCE.contains(reset)
            && SOURCE.contains(isolated)
            && SOURCE.contains(production)
            && SOURCE.contains(arm)
            && SOURCE.contains(closure)
    );
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
        SOURCE.replacen(reset, "                    other.reset_state();", 1),
        SOURCE.replacen(reset, "", 1),
        SOURCE.replacen(reset, &format!("{reset}\n{reset}"), 1),
        SOURCE.replacen(reset, &format!("                    #[cfg(any())]\n{reset}"), 1),
        SOURCE.replacen(isolated, "            let t = bench_support::forward_prefill_gdn_isolated_chunk(other, chunk, start)?;", 1),
        SOURCE.replacen(production, "            bench_support::forward_prefill_production_chunk(other, chunk, pos)?;", 1),
        SOURCE.replacen(production, "", 1),
        SOURCE.replacen(production, &format!("{production}\n{production}"), 1),
        SOURCE.replacen(arm, "", 1),
        SOURCE.replacen(arm, &format!("{arm}\n{arm}"), 1),
        SOURCE.replacen(arm, "                    let deferred = || run_once_production(&mut state, &tokens);", 1),
        SOURCE.replacen(arm, &format!("{arm}\n                    consume!(run_once_production);"), 1),
        SOURCE.replacen(closure, "let run_once = move |state: &mut MetalQwen35State,", 1),
        SOURCE.replacen(guard, "", 1).replacen(closure, &format!("{guard}{closure}"), 1),
    ] { assert!(check(&mutated).is_err()); }
}
