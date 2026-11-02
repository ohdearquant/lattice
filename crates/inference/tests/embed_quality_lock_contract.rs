//! Selected Qwen work lifetime contract for the embedding quality example.

use syn::parse::Parser;
use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_embed_quality.rs");

fn path_is(expr: &Expr, names: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == names.len()
        && path.path.segments.iter().zip(names).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
        })
}

fn cfg_backfill_metal(attrs: &[syn::Attribute]) -> bool {
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
        && pair(&args[0], "feature", "backfill")
        && pair(&args[1], "target_os", "macos")
        && pair(&args[2], "feature", "metal-gpu")
}

fn cfg_backfill_nonmetal(attrs: &[syn::Attribute]) -> bool {
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
    if args.len() != 2
        || !matches!(&args[0], Meta::NameValue(nv) if nv.path.is_ident("feature")
        && matches!(&nv.value, Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == "backfill")))
    {
        return false;
    }
    let Meta::List(not) = &args[1] else {
        return false;
    };
    if !not.path.is_ident("not") {
        return false;
    }
    let Ok(Meta::List(inner)) = not.parse_args::<Meta>() else {
        return false;
    };
    if !inner.path.is_ident("all") {
        return false;
    }
    let Ok(pairs) = inner
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
    else {
        return false;
    };
    let pair = |meta: &Meta, key: &str, value: &str| {
        matches!(meta, Meta::NameValue(nv)
        if nv.path.is_ident(key) && matches!(&nv.value, Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == value)))
    };
    pairs.len() == 2
        && pair(&pairs[0], "target_os", "macos")
        && pair(&pairs[1], "feature", "metal-gpu")
}

fn cfg_backfill(attrs: &[syn::Attribute]) -> bool {
    let [attr] = attrs else { return false };
    attr.path().is_ident("cfg")
        && matches!(attr.parse_args::<Meta>(), Ok(Meta::NameValue(value))
        if value.path.is_ident("feature") && matches!(&value.value,
            Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == "backfill")))
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
        && binding.ident == "_gpu_guard"
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

fn direct_run(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Expr(Expr::Call(call), Some(_)) if call.attrs.is_empty()
        && call.args.is_empty() && path_is(&call.func, &["run_bench"]))
}

fn qwen_binding(local: &syn::Local) -> bool {
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::MethodCall(unwrap) = init.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = unwrap.receiver.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "qwen_model"
        && binding.mutability.is_none()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && unwrap.method == "unwrap"
        && unwrap.attrs.is_empty()
        && unwrap.args.is_empty()
        && call.attrs.is_empty()
        && call.args.len() == 1
        && path_is(
            &call.func,
            &["lattice_inference", "QwenModel", "from_directory"],
        )
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "qwen_model" | "_gpu_guard" | "gpu_test_lock" | "run_bench"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|suffix| {
                    let name: String = suffix
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    matches!(name.as_str(), "qwen_model" | "_gpu_guard")
                })
            })
        }
        _ => false,
    })
}

fn gpu_print(mac: &syn::Macro) -> bool {
    if !mac.path.is_ident("println") {
        return false;
    }
    let parser = syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated;
    let Ok(args) = parser.parse2(mac.tokens.clone()) else {
        return false;
    };
    if args.len() != 2 {
        return false;
    }
    matches!(&args[0], Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == "  Qwen3 GPU: {}\n"))
        && matches!(&args[1], Expr::MethodCall(call) if call.method == "has_gpu"
            && call.attrs.is_empty() && call.args.is_empty() && path_is(&call.receiver, &["qwen_model"]))
}

fn direct_encode(expr: &Expr, arg: &str) -> bool {
    matches!(expr, Expr::MethodCall(call) if call.method == "encode" && call.attrs.is_empty()
        && path_is(&call.receiver, &["qwen_model"]) && call.args.len() == 1 && path_is(&call.args[0], &[arg]))
}

fn immediate_map_collect(expr: &syn::ExprMethodCall) -> bool {
    if expr.method != "collect" || !expr.attrs.is_empty() || !expr.args.is_empty() {
        return false;
    }
    let Expr::MethodCall(map) = expr.receiver.as_ref() else {
        return false;
    };
    if map.method != "map" || !map.attrs.is_empty() || map.args.len() != 1 {
        return false;
    }
    let Expr::MethodCall(iter) = map.receiver.as_ref() else {
        return false;
    };
    if iter.method != "iter"
        || !iter.attrs.is_empty()
        || !iter.args.is_empty()
        || !path_is(&iter.receiver, &["contents"])
    {
        return false;
    }
    let Some(Expr::Closure(closure)) = map.args.first() else {
        return false;
    };
    if !closure.attrs.is_empty()
        || closure.capture.is_some()
        || closure.asyncness.is_some()
        || closure.inputs.len() != 1
        || !matches!(closure.inputs.first(), Some(Pat::Ident(arg)) if arg.ident == "text"
            && arg.by_ref.is_none() && arg.mutability.is_none() && arg.subpat.is_none())
    {
        return false;
    }
    matches!(closure.body.as_ref(), Expr::MethodCall(unwrap) if unwrap.method == "unwrap"
        && unwrap.attrs.is_empty() && unwrap.args.is_empty() && direct_encode(&unwrap.receiver, "text"))
}

#[derive(Default)]
struct Work {
    model_bindings: usize,
    constructors: usize,
    direct: usize,
    mapped: usize,
    prints: usize,
    deferred: usize,
    bad: bool,
}

impl<'ast> Visit<'ast> for Work {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if matches!(&local.pat, Pat::Ident(binding) if binding.ident == "qwen_model") {
            self.model_bindings += 1;
            if !qwen_binding(local) || self.deferred != 0 {
                self.bad = true;
            }
            if let Some(init) = &local.init {
                self.visit_expr(&init.expr);
            }
            return;
        }
        visit::visit_local(self, local);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        if pat.ident == "_gpu_guard" || pat.ident == "qwen_model" {
            self.bad = true;
        }
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.segments.iter().any(|part| {
            matches!(
                part.ident.to_string().as_str(),
                "qwen_model" | "_gpu_guard" | "gpu_test_lock" | "run_bench"
            )
        }) {
            self.bad = true;
        }
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(
            &call.func,
            &["lattice_inference", "QwenModel", "from_directory"],
        ) {
            self.constructors += 1;
            if self.deferred != 0 || !call.attrs.is_empty() {
                self.bad = true;
            }
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if immediate_map_collect(call) {
            if self.deferred != 0 {
                self.bad = true;
            }
            self.mapped += 1;
            return;
        }
        if call.method == "encode" && path_is(&call.receiver, &["qwen_model"]) {
            if self.deferred != 0
                || call.args.len() != 1
                || !path_is(&call.args[0], &["query"])
                || !call.attrs.is_empty()
            {
                self.bad = true;
            }
            self.direct += 1;
            for arg in &call.args {
                self.visit_expr(arg);
            }
            return;
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
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        if attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr") {
            self.bad = true;
        }
        visit::visit_attribute(self, attr);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if gpu_print(mac) {
            self.prints += 1;
        } else if sensitive(mac.tokens.clone()) {
            self.bad = true;
        } else {
            visit::visit_macro(self, mac);
        }
    }
}

#[derive(Default)]
struct OtherUses {
    bad: bool,
}
impl<'ast> Visit<'ast> for OtherUses {
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.segments.iter().any(|part| {
            matches!(
                part.ident.to_string().as_str(),
                "run_bench" | "gpu_test_lock" | "qwen_model"
            )
        }) {
            self.bad = true;
        }
        visit::visit_expr_path(self, path);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if sensitive(mac.tokens.clone()) {
            self.bad = true;
        }
        visit::visit_macro(self, mac);
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source parse")?;
    let metal_mains: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(f) if f.sig.ident == "main" && cfg_backfill_metal(&f.attrs) => Some(f),
            _ => None,
        })
        .collect();
    let [main] = metal_mains.as_slice() else {
        return Err("Metal main");
    };
    if main.sig.asyncness.is_some()
        || main.block.stmts.len() != 2
        || !guard(&main.block.stmts[0])
        || !direct_run(&main.block.stmts[1])
    {
        return Err("guarded entry");
    }
    let cpu_mains: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(f) if f.sig.ident == "main" && cfg_backfill_nonmetal(&f.attrs) => Some(f),
            _ => None,
        })
        .collect();
    let [cpu_main] = cpu_mains.as_slice() else {
        return Err("non-Metal main");
    };
    if cpu_main.block.stmts.len() != 1 || !direct_run(&cpu_main.block.stmts[0]) {
        return Err("non-Metal entry");
    }
    let runs: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(f) if f.sig.ident == "run_bench" && cfg_backfill(&f.attrs) => Some(f),
            _ => None,
        })
        .collect();
    let [run] = runs.as_slice() else {
        return Err("run_bench");
    };
    if file
        .items
        .iter()
        .filter(|item| matches!(item, Item::Fn(f) if f.sig.ident == "run_bench"))
        .count()
        != 1
    {
        return Err("run_bench declarations");
    }
    if run.sig.asyncness.is_some() {
        return Err("async entry");
    }
    let mut other = OtherUses::default();
    for item in &file.items {
        match item {
            Item::Fn(f)
                if f.sig.ident == "run_bench"
                    || (f.sig.ident == "main"
                        && (cfg_backfill_metal(&f.attrs) || cfg_backfill_nonmetal(&f.attrs))) => {}
            _ => other.visit_item(item),
        }
    }
    if other.bad {
        return Err("unreviewed entry or Qwen work");
    }
    let mut work = Work::default();
    work.visit_block(&run.block);
    if work.bad
        || work.model_bindings != 1
        || work.constructors != 1
        || work.direct != 1
        || work.mapped != 1
        || work.prints != 1
    {
        return Err("Qwen work");
    }
    Ok(())
}

#[test]
fn embed_quality_qwen_work_remains_under_the_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn embed_quality_contract_rejects_mutations() {
    let changes = [
        (
            "let _gpu_guard = lattice_inference::measurement::gpu_test_lock();",
            "let _gpu_guard = ();",
        ),
        (
            "let _gpu_guard = lattice_inference::measurement::gpu_test_lock();",
            "let _gpu_guard = lattice_inference::measurement::gpu_test_lock(); drop(_gpu_guard);",
        ),
        (
            "    run_bench();",
            "    let later = || run_bench(); later();",
        ),
        (
            ".map(|text| qwen_model.encode(text).unwrap())",
            ".map(move |text| qwen_model.encode(text).unwrap())",
        ),
        (
            ".map(|text| qwen_model.encode(text).unwrap())",
            ".map(|text| other.encode(text).unwrap())",
        ),
        ("qwen_model.encode(query)", "other.encode(query)"),
        (
            "    let contents: Vec<&str>",
            "    #[cfg(any())] let contents: Vec<&str>",
        ),
    ];
    for (old, new) in changes {
        assert!(SOURCE.contains(old), "missing mutation anchor: {old}");
        assert!(
            check(&SOURCE.replacen(old, new, 1)).is_err(),
            "accepted mutation: {old}"
        );
    }
    let alias = SOURCE.replace(
        "let qwen_query = qwen_model.encode(query).unwrap();",
        "let alias = &qwen_model; let qwen_query = alias.encode(query).unwrap();",
    );
    assert!(check(&alias).is_err(), "Qwen alias was accepted");
}
