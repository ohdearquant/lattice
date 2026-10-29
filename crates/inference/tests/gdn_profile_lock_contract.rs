//! Source contract for reset work in two Metal GDN profiling examples.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const DECODE: &str = include_str!("../examples/decode_profile.rs");
const GDN: &str = include_str!("../examples/bench_gdn_state.rs");
const ALLOCATION: &str = include_str!("../examples/bench_metal_forward_allocation.rs");

fn path_is(expr: &Expr, segments: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == segments.len()
        && path
            .path
            .segments
            .iter()
            .zip(segments)
            .all(|(part, expected)| {
                part.ident == *expected && matches!(part.arguments, syn::PathArguments::None)
            })
}

fn metal_cfg(attrs: &[syn::Attribute], counters: bool) -> bool {
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
    if args.len() != if counters { 3 } else { 2 } {
        return false;
    }
    let is_pair = |meta: &Meta, key: &str, value: &str| match meta {
        Meta::NameValue(pair) if pair.path.is_ident(key) => matches!(&pair.value,
            Expr::Lit(expr) if matches!(&expr.lit, syn::Lit::Str(s) if s.value() == value)),
        _ => false,
    };
    is_pair(&args[0], "target_os", "macos")
        && is_pair(&args[1], "feature", "metal-gpu")
        && (!counters || is_pair(&args[2], "feature", "gdn-state-counters"))
}

fn direct_guard(stmt: &Stmt) -> bool {
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
        && binding.mutability.is_none()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && call.attrs.is_empty()
        && call.args.is_empty()
        && path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        )
}

#[derive(Default)]
struct Constructors {
    q4: usize,
    f16: usize,
}
impl<'ast> Visit<'ast> for Constructors {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]) {
            self.q4 += 1;
        }
        if path_is(&call.func, &["MetalQwen35State", "new"]) {
            self.f16 += 1;
        }
        visit::visit_expr_call(self, call);
    }
}

fn state_binding(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Type(typed) = &local.pat else {
        return false;
    };
    let Pat::Ident(binding) = typed.pat.as_ref() else {
        return false;
    };
    let syn::Type::Path(ty) = typed.ty.as_ref() else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::If(branch) = init.expr.as_ref() else {
        return false;
    };
    let Some((_, otherwise)) = &branch.else_branch else {
        return false;
    };
    let Expr::Block(otherwise) = otherwise.as_ref() else {
        return false;
    };
    if !local.attrs.is_empty()
        || !typed.attrs.is_empty()
        || !branch.attrs.is_empty()
        || !otherwise.attrs.is_empty()
        || binding.ident != "state"
        || binding.mutability.is_none()
        || binding.by_ref.is_some()
        || binding.subpat.is_some()
        || ty.qself.is_some()
        || !ty.path.is_ident("MetalQwen35State")
        || init.diverge.is_some()
        || !path_is(&branch.cond, &["is_q4"])
    {
        return false;
    }
    let mut q4 = Constructors::default();
    q4.visit_block(&branch.then_branch);
    let mut f16 = Constructors::default();
    f16.visit_block(&otherwise.block);
    q4.q4 == 1 && q4.f16 == 0 && f16.q4 == 0 && f16.f16 == 1
}

fn captures_sensitive_format(s: &str) -> bool {
    s.split('{').skip(1).any(|suffix| {
        let name: String = suffix
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
        name == "state" || name == "_gpu_lock"
    })
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "state"
                | "_gpu_lock"
                | "gpu_test_lock"
                | "reset_state"
                | "forward_prefill"
                | "forward_step"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string())
                .is_ok_and(|s| captures_sensitive_format(&s.value()))
        }
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    state_bindings: usize,
    guard_bindings: usize,
    guard_calls: usize,
    run_calls: usize,
    state_paths: usize,
    reset: usize,
    prefill: usize,
    step: usize,
    report: usize,
    take_report: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for Work {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attributes += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.state_bindings += usize::from(pat.ident == "state");
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.is_ident("state") {
            self.state_paths += 1;
        }
        if path.path.is_ident("_gpu_lock") {
            self.bad = true;
        }
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ) {
            self.guard_calls += 1;
        }
        if path_is(&call.func, &["run"]) {
            self.run_calls += 1;
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let direct = path_is(&call.receiver, &["state"]);
        let work_method = call.method == "reset_state"
            || call.method == "forward_prefill"
            || call.method == "forward_step";
        if direct {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            match call.method.to_string().as_str() {
                "reset_state" => self.reset += 1,
                "forward_prefill" => self.prefill += 1,
                "forward_step" => self.step += 1,
                "gdn_state_traffic_report" => self.report += 1,
                "take_gdn_state_traffic_report" => self.take_report += 1,
                _ => self.bad = true,
            }
        } else if work_method {
            self.bad = true;
        }
        visit::visit_expr_method_call(self, call);
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
        self.bad |= sensitive(mac.tokens.clone());
        visit::visit_macro(self, mac);
    }
}

fn check(source: &str, counters: bool) -> Result<(), &'static str> {
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
    let runs = find("run");
    let mains = find("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main");
    };
    if !metal_cfg(&run.attrs, counters)
        || run.sig.asyncness.is_some()
        || main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
    {
        return Err("entry cfg changed");
    }
    let entry = main.block.stmts.iter().filter(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs, counters) && path_is(&call.func, &["run"]) && call.args.is_empty())).count();
    if entry != 1 {
        return Err("expected direct Metal entry call");
    }
    let guards = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect::<Vec<_>>();
    let states = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| state_binding(stmt))
        .collect::<Vec<_>>();
    let ([(g, _)], [(s, _)]) = (guards.as_slice(), states.as_slice()) else {
        return Err("guard or typed state missing");
    };
    if g >= s {
        return Err("guard must precede state");
    }
    let mut all = Work::default();
    for item in &file.items {
        if let Item::Fn(function) = item {
            all.visit_block(&function.block);
        } else {
            all.visit_item(item);
        }
    }
    let mut inside = Work::default();
    inside.visit_block(&run.block);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*g] {
        before.visit_stmt(stmt);
    }
    let reports = usize::from(counters);
    let takes = if counters { 2 } else { 0 };
    let paths = 4 + reports + takes;
    if all.bad
        || inside.bad
        || inside.attributes != 0
        || all.guard_calls != 1
        || all.guard_bindings != 1
        || inside.guard_bindings != 1
        || all.state_bindings != 1
        || inside.state_bindings != 1
        || all.state_paths != paths
        || inside.state_paths != paths
        || all.reset != 1
        || inside.reset != 1
        || all.prefill != 1
        || inside.prefill != 1
        || all.step != 2
        || inside.step != 2
        || all.report != reports
        || inside.report != reports
        || all.take_report != takes
        || inside.take_report != takes
        || before.state_bindings != 0
        || before.reset != 0
        || before.prefill != 0
        || before.step != 0
        || all.run_calls != 1
    {
        return Err("unreviewed guard, state or work use");
    }
    Ok(())
}

#[test]
fn profilers_keep_resets_and_forward_work_under_guard() {
    assert_eq!(check(DECODE, false), Ok(()));
    assert_eq!(check(GDN, true), Ok(()));
}

#[test]
fn profiler_contract_rejects_real_source_mutations() {
    for (source, counters) in [(DECODE, false), (GDN, true)] {
        let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
        let reset = "    state.reset_state();";
        let prefill = "state.forward_prefill(&prompt_ids)";
        assert!(source.contains(guard) && source.contains(reset) && source.contains(prefill));
        for mutated in [
            source.replacen(guard, "", 1),
            source.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
            source.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
            source.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
            source.replacen(reset, "    other.reset_state();", 1),
            source.replacen(reset, "", 1),
            source.replacen(reset, &format!("{reset}\n{reset}"), 1),
            source.replacen(reset, "    let deferred = || state.reset_state();", 1),
            source.replacen(reset, &format!("    #[cfg(any())]\n{reset}"), 1),
            source.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
            source.replacen(
                reset,
                &format!("{reset}\n    println!(\"{{state:?}}\");"),
                1,
            ),
            source.replacen(prefill, "other.forward_prefill(&prompt_ids)", 1),
            source
                .replacen(guard, "", 1)
                .replacen(reset, &format!("{reset}\n{guard}"), 1),
        ] {
            assert!(check(&mutated, counters).is_err());
        }
    }
}

#[test]
fn documented_main_and_both_metal_cfg_forms_are_recognized() {
    for (source, counters) in [(DECODE, false), (GDN, true)] {
        let file = syn::parse_file(source).unwrap();
        let main = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(f) if f.sig.ident == "main" => Some(f),
                _ => None,
            })
            .unwrap();
        assert!(!main.attrs.is_empty());
        assert!(main.attrs.iter().all(|attr| attr.path().is_ident("doc")));
        assert!(main.block.stmts.iter().any(|stmt| matches!(stmt,
            Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs, counters) && path_is(&call.func, &["run"]))));
    }
    assert!(!captures_sensitive_format("{state_bytes_moved}"));
    assert!(captures_sensitive_format("{state:?}"));
}

fn allocation_state_binding(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Match(branch) = init.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = branch.expr.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && branch.attrs.is_empty()
        && call.attrs.is_empty()
        && call.args.len() == 4
        && path_is(&call.func, &["MetalQwen35State", "from_q4_dir"])
}

fn check_allocation(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let find = |name: &str| {
        file.items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(function) if function.sig.ident == name => Some(function),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let runs = find("run");
    let mains = find("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main");
    };
    if !metal_cfg(&run.attrs, false)
        || run.sig.asyncness.is_some()
        || main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
    {
        return Err("entry cfg changed");
    }
    let entry = main
        .block
        .stmts
        .iter()
        .filter(|stmt| {
            matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs, false)
            && path_is(&call.func, &["run"]) && call.args.is_empty())
        })
        .count();
    if entry != 1 {
        return Err("expected direct Metal entry");
    }
    let guards: Vec<_> = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect();
    let states: Vec<_> = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| allocation_state_binding(stmt))
        .collect();
    let ([(g, _)], [(s, _)]) = (guards.as_slice(), states.as_slice()) else {
        return Err("guard or allocation state missing");
    };
    if g >= s {
        return Err("guard must precede state");
    }
    let mut all = Work::default();
    for item in &file.items {
        if let Item::Fn(function) = item {
            all.visit_block(&function.block);
        } else {
            all.visit_item(item);
        }
    }
    let mut inside = Work::default();
    inside.visit_block(&run.block);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*g] {
        before.visit_stmt(stmt);
    }
    let mut constructors = Constructors::default();
    constructors.visit_file(&file);
    if all.bad
        || inside.bad
        || inside.attributes != 0
        || all.guard_calls != 1
        || all.guard_bindings != 1
        || inside.guard_bindings != 1
        || all.state_bindings != 1
        || inside.state_bindings != 1
        || all.state_paths != 3
        || inside.state_paths != 3
        || all.reset != 1
        || inside.reset != 1
        || all.prefill != 0
        || inside.prefill != 0
        || all.step != 2
        || inside.step != 2
        || all.run_calls != 1
        || before.state_bindings != 0
        || before.reset != 0
        || before.step != 0
        || constructors.q4 != 1
        || constructors.f16 != 0
    {
        return Err("unreviewed guard, state or allocation work");
    }
    Ok(())
}

#[test]
fn allocation_profiler_keeps_reset_and_forward_work_under_guard() {
    assert_eq!(check_allocation(ALLOCATION), Ok(()));
}

#[test]
fn allocation_profiler_contract_rejects_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "    state.reset_state();";
    assert!(ALLOCATION.contains(guard) && ALLOCATION.contains(reset));
    for changed in [
        ALLOCATION.replacen(guard, "", 1),
        ALLOCATION.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
        ALLOCATION.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
        ALLOCATION.replacen(reset, "    other.reset_state();", 1),
        ALLOCATION.replacen(reset, "", 1),
        ALLOCATION.replacen(reset, &format!("{reset}\n{reset}"), 1),
        ALLOCATION.replacen(reset, "    let deferred = || state.reset_state();", 1),
        ALLOCATION.replacen(reset, &format!("    #[cfg(any())]\n{reset}"), 1),
        ALLOCATION.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
        ALLOCATION.replacen("state.forward_step(0, 1)", "other.forward_step(0, 1)", 1),
        ALLOCATION
            .replacen(guard, "", 1)
            .replacen(reset, &format!("{reset}\n{guard}"), 1),
        ALLOCATION.replacen("    run();", "    return;", 1),
    ] {
        assert!(
            check_allocation(&changed).is_err(),
            "allocation source mutation was accepted"
        );
    }
}
