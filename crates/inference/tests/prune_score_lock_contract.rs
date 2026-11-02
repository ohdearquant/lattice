//! Source-only lock lifetime contract for the CLI pruning measurement.

use proc_macro2::{TokenStream, TokenTree};
use syn::visit::{self, Visit};
use syn::{Expr, Item, Pat, Stmt};

fn source() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../serve/src/bin/lattice/prune_score.rs");
    std::fs::read_to_string(path).expect("read workspace prune-score source")
}

fn path_is(expr: &Expr, names: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == names.len()
        && path.path.segments.iter().zip(names).all(|(segment, name)| {
            segment.ident == *name && matches!(segment.arguments, syn::PathArguments::None)
        })
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

fn sensitive(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "_gpu_guard"
                | "dense_state"
                | "pruned_state"
                | "gpu_test_lock"
                | "from_q4_dir"
                | "compute_perplexity"
                | "score_layer_importance"
        ),
        TokenTree::Group(group) => sensitive(group.stream()),
        TokenTree::Literal(literal) => syn::parse_str::<syn::LitStr>(&literal.to_string())
            .is_ok_and(|value| {
                value.value().split('{').skip(1).any(|part| {
                    let name: String = part
                        .chars()
                        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                        .collect();
                    matches!(name.as_str(), "_gpu_guard" | "dense_state" | "pruned_state")
                })
            }),
        TokenTree::Punct(_) => false,
    })
}

#[derive(Default)]
struct Work {
    deferred: usize,
    bad: bool,
    attributes: usize,
    guards: usize,
    guard_uses: usize,
    locks: usize,
    dense_bindings: usize,
    pruned_bindings: usize,
    dense_paths: usize,
    pruned_paths: usize,
    dense_ppl: usize,
    pruned_ppl: usize,
    score: usize,
    drops: usize,
    constructors: usize,
}

impl<'ast> Visit<'ast> for Work {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attributes += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.guards += usize::from(pat.ident == "_gpu_guard");
        self.dense_bindings += usize::from(pat.ident == "dense_state");
        self.pruned_bindings += usize::from(pat.ident == "pruned_state");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.guard_uses += usize::from(path.path.is_ident("_gpu_guard"));
        self.dense_paths += usize::from(path.path.is_ident("dense_state"));
        self.pruned_paths += usize::from(path.path.is_ident("pruned_state"));
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ) {
            self.locks += 1;
        }
        if path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]) {
            self.constructors += 1;
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
        }
        if path_is(&call.func, &["drop"])
            && call.args.len() == 1
            && path_is(&call.args[0], &["dense_state"])
        {
            self.drops += 1;
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let dense = path_is(&call.receiver, &["dense_state"]);
        let pruned = path_is(&call.receiver, &["pruned_state"]);
        if (call.method == "compute_perplexity" || call.method == "score_layer_importance")
            && !dense
            && !pruned
        {
            self.bad = true;
        }
        if dense || pruned {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            match (
                dense,
                pruned,
                call.method.to_string().as_str(),
                call.args.len(),
            ) {
                (true, false, "compute_perplexity", 2) => self.dense_ppl += 1,
                (false, true, "compute_perplexity", 2) => self.pruned_ppl += 1,
                (true, false, "score_layer_importance", 2) => self.score += 1,
                _ => self.bad = true,
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
    fn visit_item(&mut self, item: &'ast Item) {
        self.deferred += 1;
        visit::visit_item(self, item);
        self.deferred -= 1;
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(mac.tokens.clone());
    }
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
    let Expr::Try(result) = init.expr.as_ref() else {
        return false;
    };
    let Expr::MethodCall(map_err) = result.expr.as_ref() else {
        return false;
    };
    let Expr::Call(constructor) = map_err.receiver.as_ref() else {
        return false;
    };
    let mut work = Work::default();
    work.visit_expr(&init.expr);
    local.attrs.is_empty()
        && result.attrs.is_empty()
        && map_err.attrs.is_empty()
        && map_err.method == "map_err"
        && map_err.args.len() == 1
        && constructor.attrs.is_empty()
        && constructor.args.len() == 4
        && path_is(&constructor.func, &["MetalQwen35State", "from_q4_dir"])
        && binding.ident == name
        && binding.by_ref.is_none()
        && binding.mutability.is_some()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && work.constructors == 1
        && !work.bad
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source parse")?;
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "run_metal" => Some(function),
            _ => None,
        })
        .collect();
    let [function] = functions.as_slice() else {
        return Err("Metal function");
    };
    if function.sig.asyncness.is_some() {
        return Err("Metal function");
    }
    let Some(guard_index) = function.block.stmts.iter().position(guard) else {
        return Err("function guard");
    };
    let mut before = Work::default();
    for stmt in &function.block.stmts[..guard_index] {
        before.visit_stmt(stmt);
    }
    if before.constructors != 0 || before.dense_paths != 0 || before.pruned_paths != 0 {
        return Err("work before guard");
    }
    for name in ["dense_state", "pruned_state"] {
        if function
            .block
            .stmts
            .iter()
            .filter(|stmt| state_binding(stmt, name))
            .count()
            != 1
        {
            return Err("measured work");
        }
    }
    let mut work = Work::default();
    work.visit_block(&function.block);
    if work.bad
        || work.attributes != 0
        || work.guards != 1
        || work.guard_uses != 0
        || work.locks != 1
    {
        return Err("guard lifetime");
    }
    if work.constructors != 2
        || work.dense_bindings != 1
        || work.pruned_bindings != 1
        || work.dense_paths != 3
        || work.pruned_paths != 1
        || work.dense_ppl != 1
        || work.pruned_ppl != 1
        || work.score != 1
        || work.drops != 1
    {
        return Err("measured work");
    }
    Ok(())
}

#[test]
fn pruning_measurement_keeps_its_guard() {
    let source = source();
    assert_eq!(check(&source), Ok(()));
    let owner_imports = source
        .replace(
            "lattice_inference::tokenizer::bpe::BpeTokenizer",
            "lattice_tokenizer::BpeTokenizer",
        )
        .replace(
            "lattice_inference::tokenizer::common::Tokenizer",
            "lattice_tokenizer::Tokenizer",
        );
    assert_eq!(check(&owner_imports), Ok(()));
}

#[test]
fn pruning_guard_contract_rejects_source_mutations() {
    let source = source();
    let mutations = [
        (
            "let _gpu_guard = lattice_inference::measurement::gpu_test_lock();",
            "let _gpu_guard = ();",
            "function guard",
        ),
        (
            "validate_args(args)?;",
            "drop(_gpu_guard); validate_args(args)?;",
            "guard lifetime",
        ),
        (
            "validate_args(args)?;",
            "let _gpu_guard = (); validate_args(args)?;",
            "guard lifetime",
        ),
        (
            "validate_args(args)?;",
            "let captured = _gpu_guard; validate_args(args)?;",
            "guard lifetime",
        ),
        (
            "let dense_report = dense_state",
            "let leaked = &dense_state; let dense_report = dense_state",
            "measured work",
        ),
        (
            ".score_layer_importance(&calibration_prompts, args.prune_layers)",
            ".other_score(&calibration_prompts, args.prune_layers)",
            "guard lifetime",
        ),
        (
            "let dense_report = dense_state",
            "#[cfg(any())] let dense_report = dense_state",
            "guard lifetime",
        ),
        (
            "let dense_report = dense_state",
            "let f = || dense_state.compute_perplexity(&validation.tokens, &ppl_cfg); let dense_report = dense_state",
            "guard lifetime",
        ),
        (
            "let dense_report = dense_state",
            "opaque!(dense_state); let dense_report = dense_state",
            "guard lifetime",
        ),
        (
            "let pruned_report = pruned_state",
            "let pruned_report = dense_state",
            "measured work",
        ),
        ("drop(dense_state);", "drop(&dense_state);", "measured work"),
        (
            "MetalQwen35State::from_q4_dir(&args.q4_dir",
            "MetalQwen35State::other(&args.q4_dir",
            "measured work",
        ),
    ];
    let lock = "let _gpu_guard = lattice_inference::measurement::gpu_test_lock();";
    let anchor = "let dense_report = dense_state";
    assert!(source.contains(lock) && source.contains(anchor));
    let insertion = [lock, "\n    ", anchor].concat();
    let late = source.replacen(lock, "", 1).replacen(anchor, &insertion, 1);
    assert!(syn::parse_file(&late).is_ok());
    assert_eq!(check(&late), Err("work before guard"));

    for (before, after, expected) in mutations {
        assert!(source.contains(before), "missing mutation anchor {before}");
        let mutant = source.replacen(before, after, 1);
        assert!(syn::parse_file(&mutant).is_ok(), "invalid mutant {before}");
        assert_eq!(check(&mutant), Err(expected), "mutation {before}");
    }
}
