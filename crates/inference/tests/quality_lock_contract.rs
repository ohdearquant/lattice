//! Source contract for Metal work in the layer-pruning quality benchmark.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_quality.rs");
const STATES: [&str; 3] = ["state", "state8", "state12"];

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
    if args.len() != 2 {
        return false;
    }
    let is_pair = |meta: &Meta, key: &str, value: &str| match meta {
        Meta::NameValue(pair) if pair.path.is_ident(key) => matches!(&pair.value,
            Expr::Lit(expr) if matches!(&expr.lit, syn::Lit::Str(s) if s.value() == value)),
        _ => false,
    };
    is_pair(&args[0], "target_os", "macos") && is_pair(&args[1], "feature", "metal-gpu")
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

fn state_index(expr: &Expr) -> Option<usize> {
    STATES.iter().position(|name| path_is(expr, &[*name]))
}

fn state_binding(stmt: &Stmt) -> Option<usize> {
    let Stmt::Local(local) = stmt else {
        return None;
    };
    let Pat::Ident(binding) = &local.pat else {
        return None;
    };
    let index = STATES.iter().position(|name| binding.ident == *name)?;
    let init = local.init.as_ref()?;
    let Expr::MethodCall(expect) = init.expr.as_ref() else {
        return None;
    };
    let Expr::Call(call) = expect.receiver.as_ref() else {
        return None;
    };
    (local.attrs.is_empty()
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && expect.attrs.is_empty()
        && expect.method == "expect"
        && expect.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == 4
        && path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]))
    .then_some(index)
}

fn sensitive_format(s: &str) -> bool {
    s.split('{').skip(1).any(|suffix| {
        let name: String = suffix
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
        name == "_gpu_lock" || STATES.contains(&name.as_str())
    })
}
fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "state"
                | "state8"
                | "state12"
                | "_gpu_lock"
                | "gpu_test_lock"
                | "reset_state"
                | "forward_prefill"
                | "score_layer_importance"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string())
                .is_ok_and(|s| sensitive_format(&s.value()))
        }
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    bindings: [usize; 3],
    paths: [usize; 3],
    reset: [usize; 3],
    prefill: [usize; 3],
    score: [usize; 3],
    drops: [usize; 3],
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
        if let Some(index) = STATES.iter().position(|name| pat.ident == *name) {
            self.bindings[index] += 1;
        }
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if let Some(index) = STATES.iter().position(|name| path.path.is_ident(*name)) {
            self.paths[index] += 1;
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
        if path_is(&call.func, &["drop"]) && call.args.len() == 1 {
            if let Some(index) = call.args.first().and_then(state_index) {
                self.bad |= self.deferred > 0 || !call.attrs.is_empty();
                self.drops[index] += 1;
            }
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let work = call.method == "reset_state"
            || call.method == "forward_prefill"
            || call.method == "score_layer_importance";
        if let Some(index) = state_index(&call.receiver) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            match call.method.to_string().as_str() {
                "reset_state" => self.reset[index] += 1,
                "forward_prefill" => self.prefill[index] += 1,
                "score_layer_importance" => self.score[index] += 1,
                _ => self.bad = true,
            }
        } else if work {
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
    let runs = find("run");
    let mains = find("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main");
    };
    if !metal_cfg(&run.attrs)
        || run.sig.asyncness.is_some()
        || main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
    {
        return Err("entry cfg changed");
    }
    let entry = main.block.stmts.iter().filter(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs) && path_is(&call.func, &["run"]) && call.args.is_empty())).count();
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
    let [(guard_index, _)] = guards.as_slice() else {
        return Err("guard missing");
    };
    let bindings = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(pos, stmt)| state_binding(stmt).map(|index| (pos, index)))
        .collect::<Vec<_>>();
    if bindings.len() != 3
        || bindings
            .iter()
            .enumerate()
            .any(|(i, (pos, index))| *pos <= *guard_index || *index != i)
    {
        return Err("unreviewed state construction");
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
    for stmt in &run.block.stmts[..*guard_index] {
        before.visit_stmt(stmt);
    }
    if all.bad
        || inside.bad
        || inside.attributes != 0
        || all.guard_calls != 1
        || all.guard_bindings != 1
        || inside.guard_bindings != 1
        || all.bindings != [1, 1, 1]
        || inside.bindings != [1, 1, 1]
        || all.paths != [4, 3, 3]
        || inside.paths != [4, 3, 3]
        || all.reset != [1, 1, 1]
        || inside.reset != [1, 1, 1]
        || all.prefill != [1, 1, 1]
        || inside.prefill != [1, 1, 1]
        || all.score != [1, 0, 0]
        || inside.score != [1, 0, 0]
        || all.drops != [1, 1, 1]
        || inside.drops != [1, 1, 1]
        || before.bindings != [0, 0, 0]
        || before.paths != [0, 0, 0]
        || all.run_calls != 1
    {
        return Err("unreviewed guard, state or work use");
    }
    Ok(())
}

#[test]
fn quality_benchmark_keeps_resets_prefill_and_scoring_under_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn quality_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "        state.reset_state();";
    let score = "        match state.score_layer_importance(&scoring_prompts, 12) {";
    assert!(SOURCE.contains(guard) && SOURCE.contains(reset) && SOURCE.contains(score));
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
        SOURCE.replacen(
            "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
            "#[cfg(any())]",
            1,
        ),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
        SOURCE.replacen(reset, "        other.reset_state();", 1),
        SOURCE.replacen(reset, "", 1),
        SOURCE.replacen(reset, &format!("{reset}\n{reset}"), 1),
        SOURCE.replacen(reset, "        let deferred = || state.reset_state();", 1),
        SOURCE.replacen(reset, &format!("        #[cfg(any())]\n{reset}"), 1),
        SOURCE.replacen(
            score,
            "        match other.score_layer_importance(&scoring_prompts, 12) {",
            1,
        ),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
        SOURCE.replacen(
            reset,
            &format!("{reset}\n        println!(\"{{state:?}}\");"),
            1,
        ),
        SOURCE.replacen("    drop(state8);", "    consume(state8);", 1),
        SOURCE
            .replacen(guard, "", 1)
            .replacen(reset, &format!("{reset}\n{guard}"), 1),
    ] {
        assert!(check(&mutated).is_err());
    }
}

#[test]
fn quality_real_documented_main_and_cfg_call_are_recognized() {
    let file = syn::parse_file(SOURCE).unwrap();
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
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs) && path_is(&call.func, &["run"]))));
}
