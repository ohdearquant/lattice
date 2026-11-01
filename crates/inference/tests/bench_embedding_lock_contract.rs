//! Source contract for Metal-capable Qwen embedding benchmark work.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_embedding.rs");

fn path_is(expr: &Expr, segments: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == segments.len()
        && path.path.segments.iter().zip(segments).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
        })
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
    expected_metal_cfg(&local.attrs)
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

fn model_constructor(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
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
        && binding.ident == "model"
        && binding.by_ref.is_none()
        && binding.mutability.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && unwrap.attrs.is_empty()
        && unwrap.method == "unwrap"
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
            "model" | "_gpu_lock" | "gpu_test_lock" | "encode"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string())
                .is_ok_and(|s| s.value().contains("{model") || s.value().contains("{_gpu_lock"))
        }
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    encodes: usize,
    other_encodes: usize,
    model_paths: usize,
    model_bindings: usize,
    guard_bindings: usize,
    guard_calls: usize,
    constructors: usize,
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
        self.model_bindings += usize::from(pat.ident == "model");
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.is_ident("model") {
            self.model_paths += 1;
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
        if path_is(
            &call.func,
            &["lattice_inference", "QwenModel", "from_directory"],
        ) {
            self.constructors += 1;
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "encode" {
            if path_is(&call.receiver, &["model"]) {
                self.bad |= self.deferred > 0 || !call.attrs.is_empty();
                self.encodes += 1;
            } else {
                self.other_encodes += 1;
            }
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

#[derive(Default)]
struct GuardBlocks<'a>(Vec<&'a syn::Block>);
impl<'ast> Visit<'ast> for GuardBlocks<'ast> {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        if block.stmts.iter().any(guard) {
            self.0.push(block);
        }
        visit::visit_block(self, block);
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let mains = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(f) if f.sig.ident == "main" => Some(f),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [main] = mains.as_slice() else {
        return Err("expected one main");
    };
    if !main.attrs.is_empty() || main.sig.asyncness.is_some() {
        return Err("main cfg changed");
    }
    let mut blocks = GuardBlocks::default();
    blocks.visit_block(&main.block);
    let [block] = blocks.0.as_slice() else {
        return Err("expected one guarded block");
    };
    let guards = block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| guard(stmt))
        .collect::<Vec<_>>();
    let constructors = block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| model_constructor(stmt))
        .collect::<Vec<_>>();
    let ([(g, _)], [(c, _)]) = (guards.as_slice(), constructors.as_slice()) else {
        return Err("guard or constructor missing");
    };
    if g >= c {
        return Err("guard must precede model construction");
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
    inside.visit_block(block);
    let mut before = Work::default();
    for stmt in &block.stmts[..*c] {
        before.visit_stmt(stmt);
    }
    if all.bad
        || inside.bad
        || inside.attributes != 1
        || all.guard_calls != 1
        || all.guard_bindings != 1
        || inside.guard_bindings != 1
        || all.model_bindings != 1
        || inside.model_bindings != 1
        || all.model_paths != 2
        || inside.model_paths != 2
        || all.encodes != 2
        || inside.encodes != 2
        || inside.other_encodes != 0
        || all.constructors != 1
        || inside.constructors != 1
        || before.encodes != 0
    {
        return Err("unreviewed guard, model, or encode use");
    }
    Ok(())
}

#[test]
fn benchmark_qwen_encode_keeps_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn benchmark_qwen_contract_rejects_real_source_mutations() {
    let guard = "            #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\n            let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let encode = "            let _ = model.encode(\"warmup\").unwrap();";
    assert!(SOURCE.contains(guard) && SOURCE.contains(encode));
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen("feature = \"metal-gpu\"", "feature = \"other\"", 1),
        SOURCE.replacen(guard, &format!("{guard}            drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(
            guard,
            &format!("{guard}            let _gpu_lock = ();\n"),
            1,
        ),
        SOURCE.replacen(
            encode,
            "            let _ = other.encode(\"warmup\").unwrap();",
            1,
        ),
        SOURCE.replacen(encode, "", 1),
        SOURCE.replacen(encode, &format!("{encode}\n{encode}"), 1),
        SOURCE.replacen(
            encode,
            &format!("{encode}\n            other.encode(\"warmup\");"),
            1,
        ),
        SOURCE.replacen(
            encode,
            "            let _ = || model.encode(\"warmup\");",
            1,
        ),
        SOURCE.replacen(
            guard,
            &format!("{guard}            consume!(_gpu_lock);\n"),
            1,
        ),
        SOURCE.replacen(
            encode,
            &format!("{encode}\n            println!(\"{{model:?}}\");"),
            1,
        ),
        SOURCE
            .replacen(guard, "", 1)
            .replacen(encode, &format!("{encode}\n{guard}"), 1),
    ] {
        assert!(check(&mutated).is_err());
    }
}
