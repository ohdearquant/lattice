//! Source contracts for the two Metal-capable benchmark-suite arms.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_suite.rs");

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

fn is_guard(stmt: &Stmt, conditional: bool) -> bool {
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
    (if conditional {
        metal_cfg(&local.attrs)
    } else {
        local.attrs.is_empty()
    }) && binding.ident == "_gpu_lock"
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

fn constructor(stmt: &Stmt, name: &str, path: &[&str], args: usize, mutable: bool) -> bool {
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
        && binding.ident == name
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && binding.mutability.is_some() == mutable
        && init.diverge.is_none()
        && expect.attrs.is_empty()
        && expect.method == "expect"
        && expect.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == args
        && path_is(&call.func, path)
}

fn sensitive(tokens: proc_macro2::TokenStream, name: &str) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => {
            let id = id.to_string();
            id == name
                || matches!(
                    id.as_str(),
                    "_gpu_lock" | "gpu_test_lock" | "reset_state" | "generate" | "encode"
                )
        }
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream(), name),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let captured: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    captured == name || captured == "_gpu_lock"
                })
            })
        }
        _ => false,
    })
}

struct Work<'a> {
    name: &'a str,
    constructor: &'a [&'a str],
    bindings: usize,
    paths: usize,
    guards: usize,
    guard_calls: usize,
    constructors: usize,
    first: usize,
    second: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}
impl<'a> Work<'a> {
    fn new(name: &'a str, constructor: &'a [&'a str]) -> Self {
        Self {
            name,
            constructor,
            bindings: 0,
            paths: 0,
            guards: 0,
            guard_calls: 0,
            constructors: 0,
            first: 0,
            second: 0,
            attributes: 0,
            deferred: 0,
            bad: false,
        }
    }
}
impl<'ast> Visit<'ast> for Work<'_> {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attributes += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.bindings += usize::from(pat.ident == self.name);
        self.guards += usize::from(pat.ident == "_gpu_lock");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.paths += usize::from(path.path.is_ident(self.name));
        self.bad |= path.path.is_ident("_gpu_lock");
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.constructors += usize::from(path_is(&call.func, self.constructor));
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if path_is(&call.receiver, &[self.name]) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            if (self.name == "metal_state" && call.method == "generate")
                || (self.name == "model" && call.method == "encode")
            {
                self.first += 1;
            } else if call.method == "reset_state" {
                self.second += 1;
            } else {
                self.bad = true;
            }
        } else if (self.name == "metal_state"
            && (call.method == "generate" || call.method == "reset_state"))
            || (self.name == "model" && call.method == "encode")
        {
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
        self.bad |= sensitive(mac.tokens.clone(), self.name);
        visit::visit_macro(self, mac);
    }
}

fn check(source: &str, embedding: bool) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let (function, name, path, args, first, second) = if embedding {
        (
            "bench_embedding",
            "model",
            &["lattice_inference", "QwenModel", "from_directory"][..],
            1,
            2,
            0,
        )
    } else {
        (
            "bench_llm_metal",
            "metal_state",
            &["MetalQwen35State", "new"][..],
            3,
            2,
            2,
        )
    };
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(f) if f.sig.ident == function => Some(f),
            _ => None,
        })
        .collect();
    let [function] = functions.as_slice() else {
        return Err("function changed");
    };
    let feature_cfg = |attrs: &[syn::Attribute]| {
        let [attr] = attrs else { return false };
        attr.path().is_ident("cfg")
            && matches!(attr.parse_args::<Meta>(), Ok(Meta::NameValue(nv))
                if nv.path.is_ident("feature") && matches!(nv.value,
                    Expr::Lit(lit) if matches!(lit.lit, syn::Lit::Str(s) if s.value() == "metal-gpu")))
    };
    if function.sig.asyncness.is_some()
        || (embedding && !function.attrs.is_empty())
        || (!embedding && !feature_cfg(&function.attrs))
    {
        return Err("function cfg or execution changed");
    }
    let guards: Vec<_> = function
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| is_guard(stmt, embedding))
        .collect();
    let bindings: Vec<_> = function
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| constructor(stmt, name, path, args, !embedding))
        .collect();
    let ([(guard, _)], [(binding, _)]) = (guards.as_slice(), bindings.as_slice()) else {
        return Err("guard or constructor changed");
    };
    if guard >= binding {
        return Err("guard must precede constructor");
    }
    let mut all = Work::new(name, path);
    all.visit_block(&function.block);
    let mut before = Work::new(name, path);
    for stmt in &function.block.stmts[..*binding] {
        before.visit_stmt(stmt);
    }
    if all.bad
        || all.attributes != usize::from(embedding)
        || all.guards != 1
        || all.guard_calls != 1
        || all.bindings != 1
        || all.paths != first + second
        || all.first != first
        || all.second != second
        || all.constructors != 1
        || before.paths != 0
        || before.first != 0
        || before.second != 0
    {
        return Err("unreviewed guard or work use");
    }
    Ok(())
}

#[test]
fn suite_arms_keep_gpu_capable_work_under_their_guards() {
    assert_eq!(check(SOURCE, false), Ok(()));
    assert_eq!(check(SOURCE, true), Ok(()));
}

#[test]
fn suite_arm_contracts_reject_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let cfg = "    #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\n";
    let metal_work = "    metal_state.reset_state();";
    let embed_work = "    let _ = model.encode(\"warmup\").unwrap();";
    assert!(
        SOURCE.contains(guard)
            && SOURCE.contains(cfg)
            && SOURCE.contains(metal_work)
            && SOURCE.contains(embed_work)
    );
    let metal_guard = format!("{guard}    let t_init = Instant::now();");
    let embed_guard = format!("{cfg}{guard}    let model =");
    assert!(SOURCE.contains(metal_guard.as_str()) && SOURCE.contains(embed_guard.as_str()));
    assert!(SOURCE.contains("#[cfg(feature = \"metal-gpu\")]\nfn bench_llm_metal()"));
    assert!(SOURCE.contains("metal_state.generate(\"Hello\""));
    for mutated in [
        SOURCE.replacen(
            "#[cfg(feature = \"metal-gpu\")]\nfn bench_llm_metal()",
            "#[cfg(any())]\nfn bench_llm_metal()",
            1,
        ),
        SOURCE.replacen(metal_guard.as_str(), "    let t_init = Instant::now();", 1),
        SOURCE.replacen(
            metal_guard.as_str(),
            &format!("{guard}    drop(_gpu_lock);\n    let t_init = Instant::now();"),
            1,
        ),
        SOURCE.replacen(
            metal_guard.as_str(),
            &format!("{guard}    let _gpu_lock = ();\n    let t_init = Instant::now();"),
            1,
        ),
        SOURCE.replacen(metal_work, "    other.reset_state();", 1),
        SOURCE.replacen(
            "metal_state.generate(\"Hello\"",
            "metal_state.encode(\"Hello\"",
            1,
        ),
        SOURCE.replacen(metal_work, &format!("    #[cfg(any())]\n{metal_work}"), 1),
        SOURCE.replacen(metal_work, "    let _ = || metal_state.reset_state();", 1),
        SOURCE.replacen(metal_work, "", 1),
        SOURCE.replacen(metal_work, &format!("{metal_work}\n{metal_work}"), 1),
        SOURCE
            .replacen(metal_guard.as_str(), "    let t_init = Instant::now();", 1)
            .replacen(metal_work, &format!("{metal_work}\n{guard}"), 1),
        SOURCE.replacen(
            metal_work,
            &format!("{metal_work}\n    println!(\"{{metal_state:?}}\");"),
            1,
        ),
    ] {
        assert!(check(&mutated, false).is_err());
    }
    for mutated in [
        SOURCE.replacen(embed_guard.as_str(), "    let model =", 1),
        SOURCE.replacen(
            embed_guard.as_str(),
            &format!("{cfg}{guard}    drop(_gpu_lock);\n    let model ="),
            1,
        ),
        SOURCE.replacen(
            embed_guard.as_str(),
            &format!("{cfg}{guard}    let _gpu_lock = ();\n    let model ="),
            1,
        ),
        SOURCE.replacen(
            embed_guard.as_str(),
            &format!("    #[cfg(any())]\n{guard}    let model ="),
            1,
        ),
        SOURCE.replacen(
            embed_work,
            "    let _ = other.encode(\"warmup\").unwrap();",
            1,
        ),
        SOURCE.replacen(embed_work, &format!("    #[cfg(any())]\n{embed_work}"), 1),
        SOURCE.replacen(embed_work, "    let _ = || model.encode(\"warmup\");", 1),
        SOURCE.replacen(embed_work, "", 1),
        SOURCE.replacen(embed_work, &format!("{embed_work}\n{embed_work}"), 1),
        SOURCE
            .replacen(embed_guard.as_str(), "    let model =", 1)
            .replacen(embed_work, &format!("{embed_work}\n{cfg}{guard}"), 1),
        SOURCE.replacen(
            embed_work,
            &format!("{embed_work}\n    println!(\"{{model:?}}\");"),
            1,
        ),
    ] {
        assert!(check(&mutated, true).is_err());
    }
}
