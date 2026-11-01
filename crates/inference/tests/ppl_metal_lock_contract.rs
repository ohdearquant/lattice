//! Source contract for the Metal perplexity measurement entry point.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Pat, Stmt};

const SOURCE: &str = include_str!("../src/bin/ppl_metal.rs");

#[derive(Default)]
struct Calls {
    lock_names: usize,
    perplexity_calls: usize,
    sensitive_macro: bool,
}

fn token_has_sensitive_ident(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(ident) => {
            matches!(
                ident.to_string().as_str(),
                "_gpu_lock" | "gpu_test_lock" | "compute_perplexity"
            )
        }
        proc_macro2::TokenTree::Group(group) => token_has_sensitive_ident(group.stream()),
        _ => false,
    })
}

impl<'ast> Visit<'ast> for Calls {
    fn visit_ident(&mut self, ident: &'ast syn::Ident) {
        if ident == "_gpu_lock" {
            self.lock_names += 1;
        }
        visit::visit_ident(self, ident);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "compute_perplexity" {
            self.perplexity_calls += 1;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.sensitive_macro |= token_has_sensitive_ident(mac.tokens.clone());
        visit::visit_macro(self, mac);
    }
}

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
            .all(|(actual, expected)| {
                actual.ident == *expected && matches!(actual.arguments, syn::PathArguments::None)
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

fn direct_perplexity(stmt: &Stmt) -> bool {
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
    let Expr::MethodCall(work) = expect.receiver.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "report"
        && binding.by_ref.is_none()
        && binding.mutability.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && expect.attrs.is_empty()
        && work.attrs.is_empty()
        && expect.method == "expect"
        && expect.args.len() == 1
        && work.method == "compute_perplexity"
        && work.args.len() == 2
        && path_is(&work.receiver, &["metal"])
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let mains: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "main" => Some(function),
            _ => None,
        })
        .collect();
    let [main] = mains.as_slice() else {
        return Err("expected one main");
    };
    if !main.attrs.is_empty() {
        return Err("main has conditional or unreviewed attributes");
    }
    let guards: Vec<_> = main
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect();
    let work: Vec<_> = main
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_perplexity(stmt))
        .collect();
    let ([(guard_index, _)], [(work_index, _)]) = (guards.as_slice(), work.as_slice()) else {
        return Err("expected direct guard and perplexity work");
    };
    if guard_index >= work_index {
        return Err("guard must precede work");
    }
    let mut calls = Calls::default();
    calls.visit_file(&file);
    if calls.lock_names != 1 || calls.perplexity_calls != 1 || calls.sensitive_macro {
        return Err("unreviewed lock or perplexity use");
    }
    Ok(())
}

#[test]
fn ppl_metal_perplexity_work_keeps_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn ppl_metal_guard_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let work = "    let report = metal.compute_perplexity(tokens, &ppl_cfg).expect(\"ppl\");";
    assert!(SOURCE.contains(guard));
    assert!(SOURCE.contains(work));
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
        SOURCE.replacen(
            work,
            "    let report = other.compute_perplexity(tokens, &ppl_cfg).expect(\"ppl\");",
            1,
        ),
        SOURCE.replacen(work, "    let report = ();", 1),
        SOURCE.replacen(
            work,
            &format!("{work}\n    let _ = metal.compute_perplexity(tokens, &ppl_cfg);"),
            1,
        ),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
        SOURCE
            .replacen(guard, "", 1)
            .replacen(work, &format!("{work}\n{guard}"), 1),
    ] {
        assert!(check(&mutated).is_err());
    }
}
