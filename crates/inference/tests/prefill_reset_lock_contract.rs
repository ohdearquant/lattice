//! Source contract for Q4/Q8 Metal prefill and reset work.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const Q4: &str = include_str!("../examples/bench_q4_prefill.rs");
const Q8: &str = include_str!("../examples/bench_q8_prefill.rs");

fn path_is(expr: &Expr, segments: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == segments.len()
        && path.path.segments.iter().zip(segments).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
        })
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

fn direct_state(stmt: &Stmt, constructor: &str, args: usize) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::MethodCall(expect) = init.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = expect.receiver.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && expect.attrs.is_empty()
        && expect.method == "expect"
        && expect.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == args
        && path_is(&call.func, &["MetalQwen35State", constructor])
}

fn expected_metal_cfg(attrs: &[syn::Attribute]) -> bool {
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
    if args.len() != 2 {
        return false;
    }
    let matches = |meta: &Meta, key: &str, value: &str| match meta {
        Meta::NameValue(pair) if pair.path.is_ident(key) => matches!(
            &pair.value,
            Expr::Lit(expr) if matches!(&expr.lit, syn::Lit::Str(s) if s.value() == value)
        ),
        _ => false,
    };
    matches(&args[0], "target_os", "macos") && matches(&args[1], "feature", "metal-gpu")
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "state" | "_gpu_lock" | "gpu_test_lock" | "forward_prefill" | "reset_state"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string())
                .is_ok_and(|s| s.value().contains("{state") || s.value().contains("{_gpu_lock"))
        }
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    prefill: usize,
    reset: usize,
    receiver_paths: usize,
    state_bindings: usize,
    guard_bindings: usize,
    guard_calls: usize,
    run_calls: usize,
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
            self.receiver_paths += 1;
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
        if call.method == "forward_prefill" || call.method == "reset_state" {
            self.bad |=
                self.deferred > 0 || !call.attrs.is_empty() || !path_is(&call.receiver, &["state"]);
            self.prefill += usize::from(call.method == "forward_prefill");
            self.reset += usize::from(call.method == "reset_state");
        }
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.deferred += 1;
        visit::visit_expr_closure(self, closure);
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

fn check(source: &str, constructor: &str, args: usize) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let find_fn = |name: &str| {
        file.items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(f) if f.sig.ident == name => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let runs = find_fn("run");
    let mains = find_fn("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main");
    };
    if !expected_metal_cfg(&run.attrs) || run.sig.asyncness.is_some() {
        return Err("run cfg changed");
    }
    let guards: Vec<_> = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect();
    let [(guard_index, _)] = guards.as_slice() else {
        return Err("expected one direct guard");
    };
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
    for stmt in &run.block.stmts[..*guard_index] {
        before.visit_stmt(stmt);
    }
    let mut entry = Work::default();
    entry.visit_block(&main.block);
    let entry_calls = main
        .block
        .stmts
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Expr(Expr::Call(call), _)
                if expected_metal_cfg(&call.attrs)
                    && path_is(&call.func, &["run"])
                    && call.args.is_empty() =>
            {
                Some(call)
            }
            _ => None,
        })
        .count();
    if !main.attrs.is_empty() || main.sig.asyncness.is_some() || entry_calls != 1 {
        return Err("Metal main entry changed");
    }
    let states = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_state(stmt, constructor, args))
        .collect::<Vec<_>>();
    let [(state_index, _)] = states.as_slice() else {
        return Err("expected reviewed state constructor");
    };
    if guard_index >= state_index {
        return Err("guard must precede state");
    }
    if all.bad
        || inside.bad
        || inside.attributes != 0
        || all.guard_calls != 1
        || all.guard_bindings != 1
        || inside.guard_bindings != 1
        || all.state_bindings != 1
        || inside.state_bindings != 1
        || all.receiver_paths != 4
        || inside.receiver_paths != 4
        || all.prefill != 2
        || inside.prefill != 2
        || all.reset != 2
        || inside.reset != 2
        || before.prefill != 0
        || before.reset != 0
        || before.state_bindings != 0
        || all.run_calls != 1
        || entry.run_calls != 1
    {
        return Err("unreviewed guard, state, work, or entrypoint use");
    }
    Ok(())
}

#[test]
fn q4_and_q8_prefill_resets_keep_the_measurement_guard() {
    assert_eq!(check(Q4, "from_q4_dir", 4), Ok(()));
    assert_eq!(check(Q8, "new", 3), Ok(()));
}

#[test]
fn prefill_contract_rejects_real_source_mutations() {
    for (source, constructor, args) in [(Q4, "from_q4_dir", 4), (Q8, "new", 3)] {
        let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
        let reset = "        state.reset_state();";
        let prefill = "        let _ = state.forward_prefill(&prompt_ids);";
        assert!(source.contains(guard) && source.contains(reset) && source.contains(prefill));
        for mutated in [
            source.replacen(guard, "", 1),
            source.replacen(
                "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
                "#[cfg(any())]",
                1,
            ),
            source.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
            source.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
            source.replacen(reset, "        other.reset_state();", 1),
            source.replacen(reset, "", 1),
            source.replacen(reset, &format!("{reset}\n{reset}"), 1),
            source.replacen(
                prefill,
                "        let _ = other.forward_prefill(&prompt_ids);",
                1,
            ),
            source.replacen(reset, "        let deferred = || state.reset_state();", 1),
            source.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
            source.replacen(
                prefill,
                &format!("{prefill}\n        println!(\"{{state:?}}\");"),
                1,
            ),
            source
                .replacen(guard, "", 1)
                .replacen(prefill, &format!("{prefill}\n{guard}"), 1),
        ] {
            assert!(check(&mutated, constructor, args).is_err());
        }
    }
}
