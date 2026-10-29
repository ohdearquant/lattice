//! Source contract for direct and wrapped Qwen Metal work in bench_metal.

use syn::parse::Parser;
use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_metal.rs");

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
        && binding.ident == "_gpu_guard"
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

fn binding(stmt: &Stmt, metal: bool) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    if !local.attrs.is_empty()
        || binding.by_ref.is_some()
        || binding.subpat.is_some()
        || init.diverge.is_some()
    {
        return false;
    }
    if metal {
        let Expr::Match(choose) = init.expr.as_ref() else {
            return false;
        };
        let Expr::Call(call) = choose.expr.as_ref() else {
            return false;
        };
        binding.ident == "metal"
            && binding.mutability.is_some()
            && choose.attrs.is_empty()
            && call.attrs.is_empty()
            && call.args.len() == 3
            && path_is(&call.func, &["MetalForwardPass", "new"])
    } else {
        let Expr::MethodCall(unwrap) = init.expr.as_ref() else {
            return false;
        };
        let Expr::Call(call) = unwrap.receiver.as_ref() else {
            return false;
        };
        binding.ident == "cpu_model"
            && binding.mutability.is_none()
            && unwrap.attrs.is_empty()
            && unwrap.method == "unwrap"
            && unwrap.args.is_empty()
            && call.attrs.is_empty()
            && call.args.len() == 1
            && path_is(&call.func, &["QwenModel", "from_directory"])
    }
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "cpu_model"
                | "metal"
                | "_gpu_guard"
                | "gpu_test_lock"
                | "encode"
                | "encode_batch"
                | "forward"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let captured: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    matches!(captured.as_str(), "cpu_model" | "metal" | "_gpu_guard")
                })
            })
        }
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    bindings: [usize; 2],
    paths: [usize; 2],
    methods: [usize; 7],
    guards: usize,
    guard_calls: usize,
    constructors: [usize; 2],
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
        self.bindings[0] += usize::from(pat.ident == "cpu_model");
        self.bindings[1] += usize::from(pat.ident == "metal");
        self.guards += usize::from(pat.ident == "_gpu_guard");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.paths[0] += usize::from(path.path.is_ident("cpu_model"));
        self.paths[1] += usize::from(path.path.is_ident("metal"));
        self.bad |= path.path.is_ident("_gpu_guard");
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.constructors[0] += usize::from(path_is(&call.func, &["QwenModel", "from_directory"]));
        self.constructors[1] += usize::from(path_is(&call.func, &["MetalForwardPass", "new"]));
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if path_is(&call.receiver, &["cpu_model"]) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            let index = match call.method.to_string().as_str() {
                "has_gpu" => 0,
                "config" => 1,
                "tokenizer" => 2,
                "cache_clear" => 3,
                "encode" => 4,
                "encode_batch" => 5,
                _ => {
                    self.bad = true;
                    0
                }
            };
            self.methods[index] += 1;
        } else if path_is(&call.receiver, &["metal"]) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty() || call.method != "forward";
            self.methods[6] += 1;
        } else if matches!(
            call.method.to_string().as_str(),
            "encode" | "encode_batch" | "forward"
        ) {
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
        if !sensitive(mac.tokens.clone()) {
            return;
        }
        if !mac.path.is_ident("println") {
            self.bad = true;
            return;
        }
        let parser = syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated;
        let Ok(expressions) = parser.parse2(mac.tokens.clone()) else {
            self.bad = true;
            return;
        };
        for expr in &expressions {
            if let Expr::Lit(lit) = expr {
                if let syn::Lit::Str(s) = &lit.lit {
                    if sensitive(proc_macro2::TokenStream::from(
                        proc_macro2::TokenTree::Literal(proc_macro2::Literal::string(&s.value())),
                    )) {
                        self.bad = true;
                    }
                }
            }
            self.visit_expr(expr);
        }
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
    let runs = find("run_bench");
    let ([main], [run]) = (mains.as_slice(), runs.as_slice()) else {
        return Err("entry changed");
    };
    if main.attrs.iter().any(|a| !a.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
        || !metal_cfg(&run.attrs)
        || run.sig.asyncness.is_some()
    {
        return Err("entry cfg changed");
    }
    if main.block.stmts.iter().filter(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs) && path_is(&call.func, &["run_bench"]) && call.args.is_empty())).count() != 1 {
        return Err("direct entry changed");
    }
    let guards = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| guard(stmt))
        .collect::<Vec<_>>();
    let models = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| binding(stmt, false))
        .collect::<Vec<_>>();
    let metals = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| binding(stmt, true))
        .collect::<Vec<_>>();
    let ([(g, _)], [(model, _)], [(metal, _)]) =
        (guards.as_slice(), models.as_slice(), metals.as_slice())
    else {
        return Err("guard or constructor changed");
    };
    if g >= model || g >= metal {
        return Err("guard must precede constructors");
    }
    let mut all = Work::default();
    all.visit_block(&run.block);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*g] {
        before.visit_stmt(stmt);
    }
    if all.bad
        || all.attributes != 0
        || all.guards != 1
        || all.guard_calls != 1
        || all.bindings != [1, 1]
        || all.paths != [12, 2]
        || all.methods != [1, 1, 1, 4, 4, 1, 2]
        || all.constructors != [1, 1]
        || before.paths != [0, 0]
        || before.bindings != [0, 0]
    {
        return Err("unreviewed guard or GPU-capable work");
    }
    Ok(())
}

#[test]
fn metal_benchmark_keeps_direct_and_qwen_work_under_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn metal_benchmark_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_guard = lattice_inference::measurement::gpu_test_lock();\n";
    let direct = "        let _ = metal.forward(&hidden_input, seq_len).unwrap();";
    let encode = "    let _ = cpu_model.encode(\"warmup\").unwrap();";
    let batch = "        let _ = cpu_model.encode_batch(&batch_refs).unwrap();";
    assert!(
        SOURCE.contains(guard)
            && SOURCE.contains(direct)
            && SOURCE.contains(encode)
            && SOURCE.contains(batch)
    );
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_guard);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu_guard = ();\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_guard);\n"), 1),
        SOURCE.replacen(
            direct,
            "        let _ = other.forward(&hidden_input, seq_len).unwrap();",
            1,
        ),
        SOURCE.replacen(
            direct,
            "        let _ = || metal.forward(&hidden_input, seq_len);",
            1,
        ),
        SOURCE.replacen(direct, "", 1),
        SOURCE.replacen(direct, &format!("{direct}\n{direct}"), 1),
        SOURCE.replacen(direct, &format!("        #[cfg(any())]\n{direct}"), 1),
        SOURCE.replacen(encode, "    let _ = other.encode(\"warmup\").unwrap();", 1),
        SOURCE.replacen(batch, "", 1),
        SOURCE.replacen(
            encode,
            &format!("{encode}\n    println!(\"{{cpu_model:?}}\");"),
            1,
        ),
        SOURCE
            .replacen(guard, "", 1)
            .replacen(encode, &format!("{encode}\n{guard}"), 1),
        SOURCE.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
    ] {
        assert!(check(&mutated).is_err());
    }
}
