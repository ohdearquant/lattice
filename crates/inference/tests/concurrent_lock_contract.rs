//! Source contract for the concurrent benchmark’s guarded helper chain.

use syn::visit::{self, Visit};
use syn::{Expr, FnArg, Item, Meta, Pat, Stmt, Type};

const SOURCE: &str = include_str!("../examples/bench_concurrent.rs");

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

fn guarded_main(main: &syn::ItemFn) -> bool {
    if main.sig.asyncness.is_some() || main.attrs.iter().any(|attr| !attr.path().is_ident("doc")) {
        return false;
    }
    let blocks = main
        .block
        .stmts
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Expr(Expr::Block(block), _) if metal_cfg(&block.attrs) => Some(block),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [block] = blocks.as_slice() else {
        return false;
    };
    let [first, Stmt::Expr(Expr::If(entry), _)] = block.block.stmts.as_slice() else {
        return false;
    };
    let Expr::Let(condition) = entry.cond.as_ref() else {
        return false;
    };
    let Expr::Call(call) = condition.expr.as_ref() else {
        return false;
    };
    guard(first)
        && entry.attrs.is_empty()
        && entry.else_branch.is_none()
        && condition.attrs.is_empty()
        && call.attrs.is_empty()
        && path_is(&call.func, &["run"])
        && call.args.is_empty()
}

fn state_binding(stmt: &Stmt, name: &str) -> bool {
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
        && binding.ident == name
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && try_expr.attrs.is_empty()
        && map_err.attrs.is_empty()
        && map_err.method == "map_err"
        && map_err.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == 4
        && path_is(&call.func, &["MetalQwen35State", "from_q4_dir"])
}

fn result_binding(stmt: &Stmt, binding_name: &str, callee: &str) -> bool {
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
        && binding.ident == binding_name
        && binding.mutability.is_none()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && try_expr.attrs.is_empty()
        && call.attrs.is_empty()
        && path_is(&call.func, &[callee])
}

fn state_param(arg: &FnArg, expected: &str) -> bool {
    let FnArg::Typed(arg) = arg else { return false };
    let Pat::Ident(name) = arg.pat.as_ref() else {
        return false;
    };
    let Type::Reference(reference) = arg.ty.as_ref() else {
        return false;
    };
    let Type::Path(ty) = reference.elem.as_ref() else {
        return false;
    };
    name.ident == expected
        && name.mutability.is_none()
        && name.by_ref.is_none()
        && name.subpat.is_none()
        && reference.mutability.is_some()
        && ty.qself.is_none()
        && ty
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "MetalQwen35State")
}

const NAMES: &[&str] = &[
    "_gpu_lock",
    "gpu_test_lock",
    "state1",
    "state2",
    "state",
    "state_a",
    "state_b",
    "run",
    "measure_single_session",
    "measure_concurrent_interleaved",
    "reset_state",
    "forward_step",
    "from_q4_dir",
];

#[derive(Default)]
struct Work {
    names: [usize; 5],
    bindings: [usize; 5],
    calls: [usize; 2],
    call_paths: [usize; 2],
    constructors: usize,
    devices: usize,
    resets: [usize; 5],
    steps: [usize; 5],
    attrs: usize,
    depth: usize,
    deferred: usize,
    bad: bool,
}
impl Work {
    fn index(expr: &Expr) -> Option<usize> {
        ["state1", "state2", "state", "state_a", "state_b"]
            .iter()
            .position(|name| path_is(expr, &[*name]))
    }
}
impl<'ast> Visit<'ast> for Work {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attrs += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        for (i, name) in ["state1", "state2", "state", "state_a", "state_b"]
            .iter()
            .enumerate()
        {
            self.bindings[i] += usize::from(pat.ident == *name);
        }
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        for (i, name) in ["state1", "state2", "state", "state_a", "state_b"]
            .iter()
            .enumerate()
        {
            self.names[i] += usize::from(path.path.is_ident(*name));
        }
        for (i, name) in ["measure_single_session", "measure_concurrent_interleaved"]
            .iter()
            .enumerate()
        {
            self.call_paths[i] += usize::from(path.path.is_ident(*name));
        }
        self.bad |= path.path.is_ident("_gpu_lock");
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.constructors += usize::from(path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]));
        self.devices += usize::from(path_is(&call.func, &["Device", "system_default"]));
        for (i, name) in ["measure_single_session", "measure_concurrent_interleaved"]
            .iter()
            .enumerate()
        {
            if path_is(&call.func, &[*name]) {
                self.calls[i] += 1;
                self.bad |= self.depth != 0
                    || self.deferred != 0
                    || !call.attrs.is_empty()
                    || call.args.len() != if i == 0 { 4 } else { 6 };
                if !call.args.is_empty() {
                    self.bad |= !matches!(&call.args[0], Expr::Reference(r) if r.mutability.is_some() && path_is(&r.expr, &["state1"]));
                }
                if i == 1 && call.args.len() > 1 {
                    self.bad |= !matches!(&call.args[1], Expr::Reference(r) if r.mutability.is_some() && path_is(&r.expr, &["state2"]));
                }
            }
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if let Some(index) = Self::index(&call.receiver) {
            self.bad |= self.depth != 0 || self.deferred != 0 || !call.attrs.is_empty();
            match call.method.to_string().as_str() {
                "reset_state" => {
                    self.resets[index] += 1;
                    self.bad |= !call.args.is_empty();
                }
                "forward_step" => {
                    self.steps[index] += 1;
                    self.bad |= call.args.len() != 2;
                }
                _ => self.bad = true,
            }
        } else if call.method == "reset_state" || call.method == "forward_step" {
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
        self.bad |= sensitive(mac.tokens.clone(), NAMES);
    }
}

#[derive(Default)]
struct Global {
    run_paths: usize,
    run_calls: usize,
    guard_calls: usize,
    resets: usize,
    steps: usize,
    constructors: usize,
    calls: [usize; 2],
    bad: bool,
}
impl<'ast> Visit<'ast> for Global {
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.run_paths += usize::from(path.path.is_ident("run"));
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.run_calls += usize::from(path_is(&call.func, &["run"]));
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.constructors += usize::from(path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]));
        for (i, name) in ["measure_single_session", "measure_concurrent_interleaved"]
            .iter()
            .enumerate()
        {
            self.calls[i] += usize::from(path_is(&call.func, &[*name]));
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.resets += usize::from(call.method == "reset_state");
        self.steps += usize::from(call.method == "forward_step");
        visit::visit_expr_method_call(self, call);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(mac.tokens.clone(), NAMES);
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
    let mains = find("main");
    let runs = find("run");
    let singles = find("measure_single_session");
    let concurrents = find("measure_concurrent_interleaved");
    let ([main], [run], [single], [concurrent]) = (
        mains.as_slice(),
        runs.as_slice(),
        singles.as_slice(),
        concurrents.as_slice(),
    ) else {
        return Err("function inventory changed");
    };
    if !guarded_main(main)
        || !metal_cfg(&run.attrs)
        || !metal_cfg(&single.attrs)
        || !metal_cfg(&concurrent.attrs)
        || run.sig.asyncness.is_some()
        || single.sig.asyncness.is_some()
        || concurrent.sig.asyncness.is_some()
    {
        return Err("guarded entry or function cfg changed");
    }
    if single
        .sig
        .inputs
        .first()
        .is_none_or(|arg| !state_param(arg, "state"))
        || concurrent
            .sig
            .inputs
            .first()
            .is_none_or(|arg| !state_param(arg, "state_a"))
        || concurrent
            .sig
            .inputs
            .iter()
            .nth(1)
            .is_none_or(|arg| !state_param(arg, "state_b"))
    {
        return Err("helper state parameters changed");
    }
    let positions = |predicate: &dyn Fn(&Stmt) -> bool| {
        run.block
            .stmts
            .iter()
            .enumerate()
            .filter(|(_, stmt)| predicate(stmt))
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    };
    let a = positions(&|stmt| state_binding(stmt, "state1"));
    let b = positions(&|stmt| state_binding(stmt, "state2"));
    let s = positions(&|stmt| result_binding(stmt, "single", "measure_single_session"));
    let c = positions(&|stmt| result_binding(stmt, "concurrent", "measure_concurrent_interleaved"));
    let ([a], [b], [s], [c]) = (a.as_slice(), b.as_slice(), s.as_slice(), c.as_slice()) else {
        return Err("state or helper binding changed");
    };
    if !(a < s && s < b && b < c) {
        return Err("state/helper order changed");
    }
    let mut main_work = Work::default();
    main_work.visit_block(&main.block);
    let mut run_work = Work::default();
    run_work.visit_block(&run.block);
    let mut one = Work::default();
    one.visit_block(&single.block);
    let mut both = Work::default();
    both.visit_block(&concurrent.block);
    let mut all = Global::default();
    all.visit_file(&file);
    if main_work.bad
        || run_work.bad
        || one.bad
        || both.bad
        || run_work.attrs != 0
        || one.attrs != 0
        || both.attrs != 0
        || run_work.bindings != [1, 1, 0, 0, 0]
        || run_work.names != [2, 1, 0, 0, 0]
        || run_work.calls != [1, 1]
        || run_work.call_paths != [1, 1]
        || run_work.constructors != 2
        || run_work.devices != 1
        || one.bindings != [0; 5]
        || one.names != [0, 0, 3, 0, 0]
        || one.resets != [0, 0, 1, 0, 0]
        || one.steps != [0, 0, 2, 0, 0]
        || both.bindings != [0; 5]
        || both.names != [0, 0, 0, 3, 3]
        || both.resets != [0, 0, 0, 1, 1]
        || both.steps != [0, 0, 0, 2, 2]
        || all.bad
        || all.run_paths != 1
        || all.run_calls != 1
        || all.guard_calls != 1
        || all.constructors != 2
        || all.calls != [1, 1]
        || all.resets != 3
        || all.steps != 6
    {
        return Err("unreviewed guard, state or helper use");
    }
    Ok(())
}

#[test]
fn concurrent_benchmark_helper_chain_keeps_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn concurrent_contract_rejects_real_source_mutations() {
    let guard = "        let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "    state.reset_state();";
    let single =
        "    let single = measure_single_session(&mut state1, total_steps, TOKEN_START_A, 0)?;";
    let concurrent = "    let concurrent = measure_concurrent_interleaved(";
    let run = "        if let Err(err) = run() {";
    assert!(
        SOURCE.contains(guard)
            && SOURCE.contains(reset)
            && SOURCE.contains(single)
            && SOURCE.contains(concurrent)
            && SOURCE.contains(run)
    );
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}        drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}        let _gpu_lock = ();\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}        consume!(_gpu_lock);\n"), 1),
        SOURCE.replacen(run, "        if let Err(err) = (|| run())() {", 1),
        SOURCE.replacen(reset, "    other.reset_state();", 1),
        SOURCE.replacen(reset, "", 1),
        SOURCE.replacen(reset, &format!("{reset}\n{reset}"), 1),
        SOURCE.replacen(reset, &format!("    #[cfg(any())]\n{reset}"), 1),
        SOURCE.replacen(single, "    let single = measure_single_session(&mut state2, total_steps, TOKEN_START_A, 0)?;", 1),
        SOURCE.replacen(single, "    let single = (|| measure_single_session(&mut state1, total_steps, TOKEN_START_A, 0))()?;", 1),
        SOURCE.replacen(single, "", 1),
        SOURCE.replacen(single, &format!("{single}\n{single}"), 1),
        SOURCE.replacen(concurrent, "    let concurrent = other_measure(", 1),
        SOURCE.replacen(single, &format!("{single}\n    consume!(measure_single_session);"), 1),
        SOURCE.replacen(guard, "", 1).replacen(run, &format!("{run}\n{guard}"), 1),
        SOURCE.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
    ] { assert!(check(&mutated).is_err()); }
}
