//! Source contract for the multimodal prefill benchmark callback.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt, Type};

const SOURCE: &str = include_str!("../src/bin/bench_vision_prefill_ab.rs");

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

fn typed_arg(pat: &Pat, name: &str, expected: &str, reference: bool) -> bool {
    let Pat::Type(typed) = pat else { return false };
    let Pat::Ident(binding) = typed.pat.as_ref() else {
        return false;
    };
    let ty = if reference {
        let Type::Reference(reference) = typed.ty.as_ref() else {
            return false;
        };
        if reference.mutability.is_none() {
            return false;
        }
        reference.elem.as_ref()
    } else {
        typed.ty.as_ref()
    };
    let Type::Path(path) = ty else { return false };
    binding.ident == name
        && binding.by_ref.is_none()
        && binding.mutability.is_none()
        && binding.subpat.is_none()
        && path.qself.is_none()
        && path.path.is_ident(expected)
}

fn arm_binding(stmt: &Stmt) -> Option<&syn::ExprClosure> {
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
        || binding.ident != "run_arm"
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
    let (Some(first), Some(second)) = (closure.inputs.first(), closure.inputs.last()) else {
        return None;
    };
    if !typed_arg(first, "state", "MetalQwen35State", true)
        || !typed_arg(second, "always_emit_head", "bool", false)
    {
        return None;
    }
    Some(closure)
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "_gpu_lock"
                | "gpu_test_lock"
                | "state"
                | "run_arm"
                | "reset_state"
                | "run_multimodal_prefill"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let captured: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    matches!(captured.as_str(), "_gpu_lock" | "state" | "run_arm")
                })
            })
        }
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    guards: usize,
    guard_calls: usize,
    states: usize,
    state_paths: usize,
    arms: usize,
    arm_paths: usize,
    constructors: usize,
    resets: usize,
    prefills: usize,
    calls_true: usize,
    calls_false: usize,
    attrs: usize,
    depth: usize,
    deferred: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for Work {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attrs += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.guards += usize::from(pat.ident == "_gpu_lock");
        self.states += usize::from(pat.ident == "state");
        self.arms += usize::from(pat.ident == "run_arm");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.state_paths += usize::from(path.path.is_ident("state"));
        self.arm_paths += usize::from(path.path.is_ident("run_arm"));
        self.bad |= path.path.is_ident("_gpu_lock");
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.constructors += usize::from(path_is(&call.func, &["MetalQwen35State", "new"]));
        if path_is(&call.func, &["bench_support", "run_multimodal_prefill"]) {
            self.prefills += 1;
            self.bad |= self.depth != 1
                || self.deferred != 0
                || !call.attrs.is_empty()
                || call.args.len() != 4
                || !path_is(&call.args[0], &["state"])
                || !matches!(&call.args[1], Expr::Reference(r) if r.mutability.is_none() && path_is(&r.expr, &["request"]))
                || !matches!(&call.args[2], Expr::Reference(r) if r.mutability.is_none() && path_is(&r.expr, &["cfg"]))
                || !path_is(&call.args[3], &["always_emit_head"]);
        }
        if path_is(&call.func, &["run_arm"]) {
            self.bad |= self.depth != 0
                || self.deferred != 0
                || !call.attrs.is_empty()
                || call.args.len() != 2;
            if call.args.len() == 2 {
                self.bad |= !matches!(&call.args[0], Expr::Reference(r) if r.mutability.is_some() && path_is(&r.expr, &["state"]));
                match &call.args[1] {
                    Expr::Lit(lit) => match &lit.lit {
                        syn::Lit::Bool(value) if value.value => self.calls_true += 1,
                        syn::Lit::Bool(_) => self.calls_false += 1,
                        _ => self.bad = true,
                    },
                    _ => self.bad = true,
                }
            }
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "reset_state" {
            self.resets += 1;
            self.bad |= !path_is(&call.receiver, &["state"])
                || self.depth != 1
                || self.deferred != 0
                || !call.attrs.is_empty()
                || !call.args.is_empty();
        } else if path_is(&call.receiver, &["state"]) {
            self.bad = true;
        }
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.depth += 1;
        visit::visit_expr_closure(self, closure);
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
    reset: usize,
    prefill: usize,
    arms: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for Global {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.reset += usize::from(call.method == "reset_state");
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.prefill += usize::from(path_is(
            &call.func,
            &["bench_support", "run_multimodal_prefill"],
        ));
        self.arms += usize::from(path_is(&call.func, &["run_arm"]));
        visit::visit_expr_call(self, call);
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
    let arms = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(i, stmt)| arm_binding(stmt).map(|arm| (i, arm)))
        .collect::<Vec<_>>();
    let ([(g, _)], [(s, _)], [(a, arm)]) = (guards.as_slice(), states.as_slice(), arms.as_slice())
    else {
        return Err("guard, state or arm changed");
    };
    if !(g < s && s < a) {
        return Err("guard and state must precede arm");
    }
    let mut all = Work::default();
    all.visit_block(&run.block);
    let mut closure = Work::default();
    closure.visit_expr_closure(arm);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*a] {
        before.visit_stmt(stmt);
    }
    let mut global = Global::default();
    global.visit_file(&file);
    if all.bad
        || all.attrs != 0
        || all.guards != 1
        || all.guard_calls != 1
        || all.states != 2
        || all.state_paths != 6
        || all.arms != 1
        || all.arm_paths != 4
        || all.constructors != 1
        || all.resets != 1
        || all.prefills != 1
        || all.calls_true != 2
        || all.calls_false != 2
        || closure.bad
        || closure.states != 1
        || closure.state_paths != 2
        || closure.resets != 1
        || closure.prefills != 1
        || closure.arm_paths != 0
        || before.arm_paths != 0
        || before.resets != 0
        || before.prefills != 0
        || global.bad
        || global.reset != 1
        || global.prefill != 1
        || global.arms != 4
    {
        return Err("unreviewed guard, closure or work use");
    }
    Ok(())
}

#[test]
fn vision_prefill_arm_and_calls_keep_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn vision_prefill_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "        state.reset_state();";
    let prefill = "            bench_support::run_multimodal_prefill(state, &request, &cfg, always_emit_head)?;";
    let arm = "        let _ = run_arm(&mut state, true)?;";
    let bind = "    let run_arm = |state: &mut MetalQwen35State, always_emit_head: bool| -> Result<f64, String> {";
    assert!(
        SOURCE.contains(guard)
            && SOURCE.contains(reset)
            && SOURCE.contains(prefill)
            && SOURCE.contains(arm)
            && SOURCE.contains(bind)
    );
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
        SOURCE.replacen(reset, "        other.reset_state();", 1),
        SOURCE.replacen(prefill, "            bench_support::run_multimodal_prefill(other, &request, &cfg, always_emit_head)?;", 1),
        SOURCE.replacen(prefill, "            Ok::<Vec<f32>, String>(vec![])?;", 1),
        SOURCE.replacen(prefill, &format!("{prefill}\n{prefill}"), 1),
        SOURCE.replacen(arm, "", 1),
        SOURCE.replacen(arm, &format!("{arm}\n{arm}"), 1),
        SOURCE.replacen(arm, "        let _ = run_arm(&mut state, false)?;", 1),
        SOURCE.replacen(arm, "        let deferred = || run_arm(&mut state, true);", 1),
        SOURCE.replacen(arm, &format!("        #[cfg(any())]\n{arm}"), 1),
        SOURCE.replacen(arm, &format!("{arm}\n        consume!(run_arm);"), 1),
        SOURCE.replacen(bind, &bind.replace("= |state", "= move |state"), 1),
        SOURCE.replacen(guard, "", 1).replacen(bind, &format!("{guard}{bind}"), 1),
    ] { assert!(check(&mutated).is_err()); }
}
