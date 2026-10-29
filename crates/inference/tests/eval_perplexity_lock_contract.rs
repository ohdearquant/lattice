//! Source contract for the two guarded Metal perplexity callers.

use proc_macro2::{TokenStream, TokenTree};
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, Item, ItemFn, Meta, Pat, Stmt, Token};

const SOURCE: &str = include_str!("../src/bin/eval_perplexity.rs");

fn path_is(expr: &Expr, expected: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == expected.len()
        && path.path.segments.iter().zip(expected).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
        })
}

fn metal_cfg(attrs: &[Attribute]) -> bool {
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
    let Ok(args) = all.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) else {
        return false;
    };
    if args.len() != 2 {
        return false;
    }
    let matches = |meta: &Meta, key: &str, expected: &str| match meta {
        Meta::NameValue(pair) if pair.path.is_ident(key) => matches!(
            &pair.value,
            Expr::Lit(expr) if matches!(&expr.lit, syn::Lit::Str(value) if value.value() == expected)
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
    metal_cfg(&local.attrs)
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

fn state_construction(stmt: &Stmt, method: &str, arguments: usize) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Match(value) = init.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = value.expr.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && value.attrs.is_empty()
        && call.attrs.is_empty()
        && call.args.len() == arguments
        && path_is(&call.func, &["MetalQwen35State", method])
}

fn tail_call(stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::Call(call), None) = stmt else {
        return false;
    };
    let Some(Expr::Reference(first)) = call.args.first() else {
        return false;
    };
    call.attrs.is_empty()
        && path_is(&call.func, &["run_on_metal_state"])
        && call.args.len() == 10
        && first.attrs.is_empty()
        && first.mutability.is_some()
        && path_is(&first.expr, &["state"])
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
    let Expr::Match(value) = init.expr.as_ref() else {
        return false;
    };
    let Expr::MethodCall(call) = value.expr.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "report"
        && binding.by_ref.is_none()
        && binding.mutability.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && value.attrs.is_empty()
        && call.attrs.is_empty()
        && call.method == "compute_perplexity"
        && call.args.len() == 2
        && path_is(&call.receiver, &["state"])
}

fn sensitive(tokens: TokenStream, check_state: bool) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(id) => {
            matches!(
                id.to_string().as_str(),
                "_gpu_lock" | "gpu_test_lock" | "run_on_metal_state" | "compute_perplexity"
            ) || (check_state && id == "state")
        }
        TokenTree::Group(group) => sensitive(group.stream(), check_state),
        TokenTree::Literal(literal) => syn::parse_str::<syn::LitStr>(&literal.to_string())
            .is_ok_and(|value| {
                value.value().split('{').skip(1).any(|part| {
                    let name: String = part
                        .chars()
                        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                        .collect();
                    name == "_gpu_lock" || (check_state && name == "state")
                })
            }),
        TokenTree::Punct(_) => false,
    })
}

#[derive(Default)]
struct Calls {
    check_state: bool,
    helper_identifiers: usize,
    helper_calls: usize,
    lock_calls: usize,
    lock_bindings: usize,
    lock_uses: usize,
    state_paths: usize,
    state_bindings: usize,
    state_methods: usize,
    perplexity_calls: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}

impl<'ast> Visit<'ast> for Calls {
    fn visit_ident(&mut self, id: &'ast syn::Ident) {
        self.helper_identifiers += usize::from(id == "run_on_metal_state");
        visit::visit_ident(self, id);
    }
    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        self.attributes += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.lock_bindings += usize::from(pat.ident == "_gpu_lock");
        self.state_bindings += usize::from(pat.ident == "state");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.lock_uses += usize::from(
            path.path
                .segments
                .iter()
                .any(|segment| segment.ident == "_gpu_lock"),
        );
        self.state_paths += usize::from(path.path.is_ident("state"));
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(&call.func, &["run_on_metal_state"]) {
            self.helper_calls += 1;
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
        }
        if path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ) {
            self.lock_calls += 1;
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if path_is(&call.receiver, &["state"]) {
            self.state_methods += 1;
            self.perplexity_calls += usize::from(call.method == "compute_perplexity");
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
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
        self.bad |= sensitive(mac.tokens.clone(), self.check_state);
    }
}

fn one_function<'a>(file: &'a syn::File, name: &str) -> Result<&'a ItemFn, &'static str> {
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .collect();
    let [function] = functions.as_slice() else {
        return Err("required function changed");
    };
    if function.sig.asyncness.is_some()
        || function
            .attrs
            .iter()
            .any(|attr| !attr.path().is_ident("doc") && !attr.path().is_ident("allow"))
    {
        return Err("function attributes changed");
    }
    Ok(function)
}

fn check_caller(function: &ItemFn, method: &str, arguments: usize) -> Result<(), &'static str> {
    let stmts = &function.block.stmts;
    let guards: Vec<_> = stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| guard(stmt))
        .collect();
    let states: Vec<_> = stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| state_construction(stmt, method, arguments))
        .collect();
    let ([(guard_index, _)], [(state_index, _)]) = (guards.as_slice(), states.as_slice()) else {
        return Err("guard or state construction changed");
    };
    if guard_index >= state_index || !stmts.last().is_some_and(tail_call) {
        return Err("guard does not enclose construction and helper");
    }
    let mut calls = Calls {
        check_state: true,
        ..Calls::default()
    };
    calls.visit_block(&function.block);
    if calls.bad
        || calls.lock_calls != 1
        || calls.lock_bindings != 1
        || calls.lock_uses != 0
        || calls.helper_calls != 1
        || calls.state_paths != 1
        || calls.state_bindings != 1
        || calls.state_methods != 0
        || calls.attributes != 1
    {
        return Err("caller guard or state escaped");
    }
    Ok(())
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let q4 = one_function(&file, "run_metal_q4")?;
    let model = one_function(&file, "run_metal_model_dir")?;
    let helper = one_function(&file, "run_on_metal_state")?;
    check_caller(q4, "from_q4_dir", 4)?;
    check_caller(model, "new", 3)?;
    if helper
        .block
        .stmts
        .iter()
        .filter(|stmt| direct_perplexity(stmt))
        .count()
        != 1
    {
        return Err("helper perplexity work changed");
    }
    let mut tail = Calls {
        check_state: true,
        ..Calls::default()
    };
    tail.visit_block(&helper.block);
    if tail.bad
        || tail.perplexity_calls != 1
        || tail.state_paths != tail.state_methods
        || tail.state_bindings != 0
        || tail.state_methods == 0
        || tail.attributes != 0
    {
        return Err("helper state escaped or was deferred");
    }
    let mut all = Calls::default();
    for item in &file.items {
        if let Item::Fn(function) = item {
            all.visit_signature(&function.sig);
            all.visit_block(&function.block);
        } else {
            all.visit_item(item);
        }
    }
    if all.bad
        || all.helper_identifiers != 3
        || all.helper_calls != 2
        || all.lock_calls != 2
        || all.lock_bindings != 2
        || all.lock_uses != 0
    {
        return Err("unreviewed helper or guard use");
    }
    Ok(())
}

fn replace_once(source: &str, before: &str, after: &str) -> String {
    assert_eq!(
        source.matches(before).count(),
        1,
        "mutation anchor {before}"
    );
    source.replacen(before, after, 1)
}

fn reject(mutant: String, expected: &str) {
    assert!(syn::parse_file(&mutant).is_ok(), "mutation must parse");
    assert_eq!(check(&mutant), Err(expected));
}

#[test]
fn both_metal_perplexity_callers_keep_a_live_guard() {
    assert_eq!(check(SOURCE), Ok(()));
    // An adapter API can change without changing the guarded call chain.
    if SOURCE.contains("state.load_lora_adapter(") {
        let renamed = SOURCE.replacen(
            "state.load_lora_adapter(",
            "state.future_adapter_method(",
            1,
        );
        assert_eq!(check(&renamed), Ok(()));
    }
}

#[test]
fn eval_perplexity_contract_rejects_guard_and_call_chain_mutations() {
    let lock = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let cfg = "    #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\n";
    let tail = "    run_on_metal_state(\n        &mut state,";
    let work = "    let report = match state.compute_perplexity(tokens, ppl_cfg) {";
    assert_eq!(SOURCE.matches(lock).count(), 2);
    assert_eq!(SOURCE.matches(tail).count(), 2);
    let first = SOURCE.replacen(lock, "", 1);
    let last = SOURCE.rreplacen(lock, "", 1);
    reject(first.clone(), "guard or state construction changed");
    reject(last, "guard or state construction changed");
    let late = first.replacen(tail, &format!("{cfg}{lock}{tail}"), 1);
    reject(late, "guard does not enclose construction and helper");
    reject(
        SOURCE.replacen(lock, &format!("    {{ {lock}    }}\n"), 1),
        "guard or state construction changed",
    );
    reject(
        SOURCE.replacen(lock, &format!("{lock}    drop(_gpu_lock);\n"), 1),
        "caller guard or state escaped",
    );
    reject(
        SOURCE.replacen(lock, &format!("{lock}    let _gpu_lock = ();\n"), 1),
        "caller guard or state escaped",
    );
    reject(
        SOURCE.replacen(
            tail,
            "    let _saved = run_on_metal_state;\n    run_on_metal_state(\n        &mut state,",
            1,
        ),
        "unreviewed helper or guard use",
    );
    let extra =
        format!("{SOURCE}\nfn extra() {{ let _ = run_on_metal_state(a,b,c,d,e,f,g,h,i,j); }}\n");
    reject(extra, "unreviewed helper or guard use");
    reject(
        replace_once(
            SOURCE,
            work,
            "    let report = match other.compute_perplexity(tokens, ppl_cfg) {",
        ),
        "helper perplexity work changed",
    );
    reject(
        replace_once(
            SOURCE,
            work,
            "    let report = match (|| state.compute_perplexity(tokens, ppl_cfg))() {",
        ),
        "helper perplexity work changed",
    );
    reject(
        replace_once(SOURCE, work, &format!("    #[cfg(any())]\n{work}")),
        "helper perplexity work changed",
    );
    reject(
        replace_once(
            SOURCE,
            work,
            &format!("    let _escaped = &mut *state;\n{work}"),
        ),
        "helper state escaped or was deferred",
    );
    reject(
        replace_once(
            SOURCE,
            work,
            &format!("    println!(\"{{state:?}}\");\n{work}"),
        ),
        "helper state escaped or was deferred",
    );
    reject(
        replace_once(
            SOURCE,
            "MetalQwen35State::from_q4_dir(q4_dir, tokenizer_path, cfg_loaded, max_cache_len)",
            "OtherState::from_q4_dir(q4_dir, tokenizer_path, cfg_loaded, max_cache_len)",
        ),
        "guard or state construction changed",
    );
}
