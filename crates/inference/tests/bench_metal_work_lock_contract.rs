//! Source contract for direct and wrapped Qwen Metal work in bench_metal.

#[path = "support/lock_contract.rs"]
mod lock_contract;

use lock_contract::path_is;
use syn::parse::Parser;
use syn::visit::Visit;
use syn::{Expr, Item, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_metal.rs");

fn metal_cfg(attrs: &[syn::Attribute]) -> bool {
    lock_contract::metal_cfg(attrs, false)
}

fn guard(stmt: &Stmt) -> bool {
    lock_contract::guard(stmt, "_gpu_guard", false)
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
    lock_contract::sensitive_tokens(
        tokens,
        &[
            "cpu_model",
            "metal",
            "_gpu_guard",
            "gpu_test_lock",
            "encode",
            "encode_batch",
            "forward",
        ],
        &["cpu_model", "metal", "_gpu_guard"],
    )
}

#[derive(Default)]
struct MetalPolicy {
    cpu_model_bindings: usize,
    metal_bindings: usize,
    cpu_model_paths: usize,
    metal_paths: usize,
    has_gpu_calls: usize,
    config_calls: usize,
    tokenizer_calls: usize,
    cache_clear_calls: usize,
    encode_calls: usize,
    encode_batch_calls: usize,
    metal_forward_calls: usize,
    guards: usize,
    guard_calls: usize,
    cpu_model_constructors: usize,
    metal_constructors: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}
type Work = lock_contract::Work<MetalPolicy>;

#[derive(Default)]
struct EntryPolicy {
    run_calls: usize,
    deferred: usize,
    bad: bool,
}

impl lock_contract::WorkPolicy for EntryPolicy {
    fn deferred(&mut self) -> &mut usize {
        &mut self.deferred
    }

    fn expr_call(&mut self, call: &syn::ExprCall) {
        if path_is(&call.func, &["run_bench"]) {
            self.run_calls += 1;
        } else {
            self.bad |= !path_is(&call.func, &["std", "process", "exit"])
                || !call.attrs.is_empty()
                || call.args.len() != 1
                || !matches!(call.args.first(), Some(Expr::Lit(lit))
                    if lit.attrs.is_empty() && matches!(&lit.lit, syn::Lit::Int(code)
                        if code.base10_parse::<i32>().is_ok_and(|code| code == 1)));
        }
    }

    fn expr_method_call(&mut self, _call: &syn::ExprMethodCall) {
        self.bad = true;
    }

    fn macro_action(&mut self, mac: &syn::Macro) -> lock_contract::MacroAction {
        self.bad |= mac
            .path
            .segments
            .last()
            .is_none_or(|name| name.ident != "eprintln")
            || syn::parse2::<syn::LitStr>(mac.tokens.clone()).is_err();
        lock_contract::MacroAction::Walk
    }
}

impl lock_contract::WorkPolicy for MetalPolicy {
    fn deferred(&mut self) -> &mut usize {
        &mut self.deferred
    }
    fn attribute(&mut self, _attr: &syn::Attribute) {
        self.attributes += 1;
    }
    fn pat_ident(&mut self, pat: &syn::PatIdent) {
        self.cpu_model_bindings += usize::from(pat.ident == "cpu_model");
        self.metal_bindings += usize::from(pat.ident == "metal");
        self.guards += usize::from(pat.ident == "_gpu_guard");
    }
    fn expr_path(&mut self, path: &syn::ExprPath) {
        self.cpu_model_paths += usize::from(path.path.is_ident("cpu_model"));
        self.metal_paths += usize::from(path.path.is_ident("metal"));
        self.bad |= path.path.is_ident("_gpu_guard");
    }
    fn expr_call(&mut self, call: &syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.cpu_model_constructors +=
            usize::from(path_is(&call.func, &["QwenModel", "from_directory"]));
        self.metal_constructors += usize::from(path_is(&call.func, &["MetalForwardPass", "new"]));
    }
    fn expr_method_call(&mut self, call: &syn::ExprMethodCall) {
        if path_is(&call.receiver, &["cpu_model"]) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            match call.method.to_string().as_str() {
                "has_gpu" => self.has_gpu_calls += 1,
                "config" => self.config_calls += 1,
                "tokenizer" => self.tokenizer_calls += 1,
                "cache_clear" => self.cache_clear_calls += 1,
                "encode" => self.encode_calls += 1,
                "encode_batch" => self.encode_batch_calls += 1,
                _ => self.bad = true,
            }
        } else if path_is(&call.receiver, &["metal"]) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty() || call.method != "forward";
            self.metal_forward_calls += 1;
        } else if matches!(
            call.method.to_string().as_str(),
            "encode" | "encode_batch" | "forward"
        ) {
            self.bad = true;
        }
    }
    fn macro_action(&mut self, mac: &syn::Macro) -> lock_contract::MacroAction {
        if !sensitive(mac.tokens.clone()) {
            return lock_contract::MacroAction::Skip;
        }
        if !mac.path.is_ident("println") {
            self.bad = true;
            return lock_contract::MacroAction::Skip;
        }
        let parser = syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated;
        let Ok(expressions) = parser.parse2(mac.tokens.clone()) else {
            self.bad = true;
            return lock_contract::MacroAction::Skip;
        };
        for expr in &expressions {
            if let Expr::Lit(lit) = expr
                && let syn::Lit::Str(s) = &lit.lit
                && sensitive(proc_macro2::TokenStream::from(
                    proc_macro2::TokenTree::Literal(proc_macro2::Literal::string(&s.value())),
                ))
            {
                self.bad = true;
            }
        }
        lock_contract::MacroAction::Expressions(expressions.into_iter().collect())
    }
}

fn check(source: &str) -> Result<(), String> {
    let file = lock_contract::check(source)?;
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
        return Err("entry changed".into());
    };
    if main.attrs.iter().any(|a| !a.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
        || !metal_cfg(&run.attrs)
        || run.sig.asyncness.is_some()
    {
        return Err("entry cfg changed".into());
    }
    if main.block.stmts.iter().filter(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs) && path_is(&call.func, &["run_bench"]) && call.args.is_empty())).count() != 1 {
        return Err("direct entry changed".into());
    }
    let mut entry = lock_contract::Work::<EntryPolicy>::default();
    entry.visit_block(&main.block);
    if entry.bad {
        return Err("unreviewed main work".into());
    }
    lock_contract::check_counts(&[("main.run_calls", entry.run_calls, 1)])?;
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
        return Err("guard or constructor changed".into());
    };
    if g >= model || g >= metal {
        return Err("guard must precede constructors".into());
    }
    let mut all = Work::default();
    all.visit_block(&run.block);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*g] {
        before.visit_stmt(stmt);
    }
    if all.bad {
        return Err("unreviewed GPU-capable work".into());
    }
    lock_contract::check_counts(&[
        ("attributes", all.attributes, 0),
        ("guards", all.guards, 1),
        ("guard_calls", all.guard_calls, 1),
        ("cpu_model_bindings", all.cpu_model_bindings, 1),
        ("metal_bindings", all.metal_bindings, 1),
        ("cpu_model_paths", all.cpu_model_paths, 12),
        ("metal_paths", all.metal_paths, 2),
        ("has_gpu_calls", all.has_gpu_calls, 1),
        ("config_calls", all.config_calls, 1),
        ("tokenizer_calls", all.tokenizer_calls, 1),
        ("cache_clear_calls", all.cache_clear_calls, 4),
        ("encode_calls", all.encode_calls, 4),
        ("encode_batch_calls", all.encode_batch_calls, 1),
        ("metal_forward_calls", all.metal_forward_calls, 2),
        ("cpu_model_constructors", all.cpu_model_constructors, 1),
        ("metal_constructors", all.metal_constructors, 1),
        ("before.cpu_model_paths", before.cpu_model_paths, 0),
        ("before.metal_paths", before.metal_paths, 0),
        ("before.cpu_model_bindings", before.cpu_model_bindings, 0),
        ("before.metal_bindings", before.metal_bindings, 0),
    ])?;
    Ok(())
}

#[test]
fn unknown_cpu_model_method_is_rejected_without_counting_as_has_gpu() {
    let unknown: Expr =
        syn::parse_str("cpu_model.unreviewed_method()").expect("fixture expression");
    let mut work = Work::default();
    work.visit_expr(&unknown);
    assert!(work.bad, "unknown cpu_model method must be rejected");
    assert_eq!(
        work.has_gpu_calls, 0,
        "unknown method must not count as has_gpu"
    );

    let known: Expr = syn::parse_str("cpu_model.has_gpu()").expect("fixture expression");
    let mut work = Work::default();
    work.visit_expr(&known);
    assert!(!work.bad, "known has_gpu method must be accepted");
    assert_eq!(work.has_gpu_calls, 1, "has_gpu method must be counted");
}

#[test]
fn metal_benchmark_keeps_direct_and_qwen_work_under_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn metal_benchmark_preserves_canonical_macro_imports() {
    for import in ["use std::vec;", "use ::std::{vec, eprintln};"] {
        let source = SOURCE.replacen("fn main() {", &format!("{import}\nfn main() {{"), 1);
        assert_eq!(check(&source), Ok(()), "{import}");
    }
}

#[test]
fn metal_benchmark_rejects_work_outside_the_guarded_entry() {
    const MAIN: &str = "fn main() {";
    const ENTRY: &str = "    run_bench();";
    assert_eq!(SOURCE.matches(MAIN).count(), 1, "main mutation anchor");
    assert_eq!(SOURCE.matches(ENTRY).count(), 1, "entry mutation anchor");
    for (name, work, expected) in [
        (
            "facade constructor",
            "let _ = lattice_inference::QwenModel::from_directory(std::path::Path::new(\"x\"));",
            "unreviewed main work",
        ),
        (
            "owner constructor",
            "let _ = lattice_model_qwen3_embedding::QwenModel::from_directory(std::path::Path::new(\"x\"));",
            "unreviewed main work",
        ),
        (
            "direct Metal constructor",
            "let _ = MetalForwardPass::new(config, weights, 1);",
            "unreviewed main work",
        ),
        ("method work", "other.forward();", "unreviewed main work"),
        ("helper work", "unreviewed_work();", "unreviewed main work"),
        (
            "macro argument work",
            "eprintln!(\"{:?}\", other.forward());",
            "unreviewed main work",
        ),
        (
            "deferred extra entry",
            "let later = || run_bench();",
            "main.run_calls: observed 2, expected 1",
        ),
    ] {
        for (location, mutated) in [
            (
                "before entry",
                SOURCE.replacen(MAIN, &format!("{MAIN}\n{work}"), 1),
            ),
            (
                "after entry",
                SOURCE.replacen(ENTRY, &format!("{ENTRY}\n{work}"), 1),
            ),
        ] {
            assert!(
                syn::parse_file(&mutated).is_ok(),
                "{name} {location} must parse"
            );
            assert_eq!(check(&mutated), Err(expected.into()), "{name} {location}");
        }
    }
}

#[test]
fn metal_benchmark_rejects_a_fake_guard_namespace() {
    let mutated = SOURCE.replacen(
        "fn main() {",
        "mod lattice_inference { pub mod measurement { pub fn gpu_test_lock() {} } }\nfn main() {",
        1,
    );
    assert!(
        syn::parse_file(&mutated).is_ok(),
        "fake guard mutant must parse"
    );
    assert_eq!(check(&mutated), Err("opaque or shadowed macro".into()));
}

#[test]
fn metal_benchmark_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_guard = lattice_inference::measurement::gpu_test_lock();\n";
    let direct = "        let _ = metal.forward(&hidden_input, seq_len).unwrap();";
    let encode = "    let _ = cpu_model.encode(\"warmup\").unwrap();";
    let batch = "        let _ = cpu_model.encode_batch(&batch_refs).unwrap();";
    for (name, anchor) in [
        ("guard", guard),
        ("direct forward", direct),
        ("encode", encode),
        ("batch", batch),
        ("main", "fn main() {"),
    ] {
        assert!(SOURCE.contains(anchor), "missing {name} mutation anchor");
    }
    for (name, mutated, expected) in [
        (
            "hidden macro GPU work",
            lock_contract::hidden_gpu_work_mutant(SOURCE, "fn run_bench(", guard),
            "opaque or shadowed macro",
        ),
        (
            "removed guard",
            SOURCE.replacen(guard, "", 1),
            "guard or constructor changed",
        ),
        (
            "dropped guard",
            SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_guard);\n"), 1),
            "unreviewed GPU-capable work",
        ),
        (
            "shadowed guard",
            SOURCE.replacen(guard, &format!("{guard}    let _gpu_guard = ();\n"), 1),
            "guards: observed 2, expected 1",
        ),
        (
            "macro guard use",
            SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_guard);\n"), 1),
            "opaque or shadowed macro",
        ),
        (
            "opaque macro without sensitive tokens",
            SOURCE
                .replacen(
                    "fn main() {",
                    "macro_rules! hidden_work { () => { lattice_inference::measurement::gpu_test_lock() }; }\nfn main() {",
                    1,
                )
                .replacen(encode, &format!("{encode}\n    hidden_work!();"), 1),
            "opaque or shadowed macro",
        ),
        (
            "shadowed formatting macro",
            SOURCE.replacen(guard, &format!("{guard}    macro_rules! println {{ () => {{}} }}\n"), 1),
            "opaque or shadowed macro",
        ),
        (
            "other forward receiver",
            SOURCE.replacen(
                direct,
                "        let _ = other.forward(&hidden_input, seq_len).unwrap();",
                1,
            ),
            "unreviewed GPU-capable work",
        ),
        (
            "deferred forward",
            SOURCE.replacen(
                direct,
                "        let _ = || metal.forward(&hidden_input, seq_len);",
                1,
            ),
            "unreviewed GPU-capable work",
        ),
        (
            "removed forward",
            SOURCE.replacen(direct, "", 1),
            "metal_paths: observed 1, expected 2",
        ),
        (
            "duplicated forward",
            SOURCE.replacen(direct, &format!("{direct}\n{direct}"), 1),
            "metal_paths: observed 3, expected 2",
        ),
        (
            "conditional forward",
            SOURCE.replacen(direct, &format!("        #[cfg(any())]\n{direct}"), 1),
            "attributes: observed 1, expected 0",
        ),
        (
            "other encode receiver",
            SOURCE.replacen(encode, "    let _ = other.encode(\"warmup\").unwrap();", 1),
            "unreviewed GPU-capable work",
        ),
        (
            "removed batch",
            SOURCE.replacen(batch, "", 1),
            "cpu_model_paths: observed 11, expected 12",
        ),
        (
            "captured cpu model",
            SOURCE.replacen(
                encode,
                &format!("{encode}\n    println!(\"{{cpu_model:?}}\");"),
                1,
            ),
            "unreviewed GPU-capable work",
        ),
        (
            "late guard",
            SOURCE
                .replacen(guard, "", 1)
                .replacen(encode, &format!("{encode}\n{guard}"), 1),
            "guard must precede constructors",
        ),
        (
            "conditional main",
            SOURCE.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
            "entry cfg changed",
        ),
    ] {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        assert_eq!(check(&mutated), Err(expected.into()), "{name}");
    }
}

#[test]
fn shared_guard_mutations_are_rejected() {
    let guard = "    let _gpu_guard = lattice_inference::measurement::gpu_test_lock();\n";
    lock_contract::assert_guard_mutations(
        SOURCE,
        guard,
        guard,
        "_gpu_guard",
        [
            "unreviewed GPU-capable work",
            "guards: observed 2, expected 1",
            "opaque or shadowed macro",
        ],
        lock_contract::GuardVector {
            expression: "QwenModel::from_directory(Path::new(\"x\"))",
            expected: "cpu_model_constructors: observed 2, expected 1",
        },
        check,
    );
}
