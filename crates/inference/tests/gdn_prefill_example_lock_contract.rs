//! Selected guard contract for the example's two GDN prefill measurement modes.

use syn::visit::{self, Visit};
use syn::{Expr, FnArg, Item, Meta, Pat, Stmt, Type};

const SOURCE: &str = include_str!("../examples/bench_gdn_prefill_ab.rs");

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
    let relevant: Vec<_> = attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("doc"))
        .collect();
    let [attr] = relevant.as_slice() else {
        return false;
    };
    if !attr.path().is_ident("cfg") {
        return false;
    }
    let Ok(meta) = attr.parse_args::<Meta>() else {
        return false;
    };
    metal_predicate(&meta)
}

fn metal_predicate(meta: &Meta) -> bool {
    let Meta::List(all) = meta else { return false };
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

fn nonmetal_cfg(attrs: &[syn::Attribute]) -> bool {
    let [attr] = attrs else { return false };
    if !attr.path().is_ident("cfg") {
        return false;
    }
    let Ok(Meta::List(not)) = attr.parse_args::<Meta>() else {
        return false;
    };
    if !not.path.is_ident("not") {
        return false;
    }
    let Ok(inner) = not.parse_args::<Meta>() else {
        return false;
    };
    metal_predicate(&inner)
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

fn state_binding(local: &syn::Local) -> bool {
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
        && call.args.len() == 3
        && path_is(&call.func, &["MetalQwen35State", "new"])
}

fn state_param(arg: &FnArg) -> bool {
    let FnArg::Typed(arg) = arg else { return false };
    let Pat::Ident(name) = arg.pat.as_ref() else {
        return false;
    };
    let Type::Reference(reference) = arg.ty.as_ref() else {
        return false;
    };
    let Type::Path(ty) = reference.elem.as_ref() else {
        return false;
    };
    name.ident == "state"
        && name.mutability.is_none()
        && name.by_ref.is_none()
        && name.subpat.is_none()
        && reference.mutability.is_some()
        && ty.qself.is_none()
        && ty
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "MetalQwen35State")
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "_gpu_lock"
                | "gpu_test_lock"
                | "state"
                | "run_one"
                | "run_interleaved"
                | "reset_state"
                | "forward_prefill"
                | "set_gdn_chunked"
                | "MetalQwen35State"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let name: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    matches!(name.as_str(), "state" | "_gpu_lock")
                })
            })
        }
        _ => false,
    })
}

fn guarded_interleaved_branch(stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::If(branch), _) = stmt else {
        return false;
    };
    let Expr::MethodCall(check) = branch.cond.as_ref() else {
        return false;
    };
    let Expr::Call(env) = check.receiver.as_ref() else {
        return false;
    };
    let [
        Stmt::Expr(Expr::Call(call), Some(_)),
        Stmt::Expr(Expr::Return(ret), Some(_)),
    ] = branch.then_branch.stmts.as_slice()
    else {
        return false;
    };
    branch.attrs.is_empty()
        && branch.else_branch.is_none()
        && check.attrs.is_empty()
        && check.method == "is_some"
        && check.args.is_empty()
        && env.attrs.is_empty()
        && path_is(&env.func, &["std", "env", "var_os"])
        && env.args.len() == 1
        && matches!(&env.args[0], Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == "BENCH_INTERLEAVED_AB"))
        && call.attrs.is_empty()
        && path_is(&call.func, &["run_interleaved"])
        && call.args.len() == 5
        && ret.attrs.is_empty()
        && ret.expr.is_none()
}

#[derive(Default)]
struct Work {
    function: String,
    guarded: bool,
    depth: usize,
    state_bindings: usize,
    constructors: usize,
    main_calls: usize,
    interleaved_calls: usize,
    helper_calls: usize,
    helper_true: usize,
    helper_false: usize,
    reset: usize,
    prefill: usize,
    mode: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for Work {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if matches!(&local.pat, Pat::Ident(binding) if binding.ident == "state") {
            self.state_bindings += 1;
            if !self.guarded
                || self.depth != 0
                || !matches!(self.function.as_str(), "run" | "run_interleaved")
                || !state_binding(local)
            {
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
        if pat.ident == "state" || pat.ident == "_gpu_lock" {
            self.bad = true;
        }
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.segments.iter().any(|part| {
            matches!(
                part.ident.to_string().as_str(),
                "state" | "_gpu_lock" | "gpu_test_lock" | "run_one" | "run_interleaved"
            )
        }) {
            self.bad = true;
        }
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(&call.func, &["run"]) {
            self.main_calls += 1;
            if self.function != "main"
                || self.depth != 0
                || !call.args.is_empty()
                || !metal_cfg(&call.attrs)
            {
                self.bad = true;
            }
            return;
        }
        if path_is(&call.func, &["run_interleaved"]) {
            self.interleaved_calls += 1;
            if self.function != "run"
                || self.guarded
                || self.depth != 0
                || !call.attrs.is_empty()
                || call.args.len() != 5
            {
                self.bad = true;
            }
            for arg in &call.args {
                self.visit_expr(arg);
            }
            return;
        }
        if path_is(&call.func, &["run_one"]) {
            self.helper_calls += 1;
            let state_arg = matches!(call.args.first(), Some(Expr::Reference(r))
                if r.mutability.is_some() && r.attrs.is_empty() && path_is(&r.expr, &["state"]));
            let prompt_arg = matches!(call.args.iter().nth(1), Some(Expr::Reference(r))
                if r.mutability.is_none() && r.attrs.is_empty() && path_is(&r.expr, &["prompt_ids"]));
            match call.args.iter().nth(2) {
                Some(Expr::Lit(lit)) if matches!(&lit.lit, syn::Lit::Bool(value) if value.value) => {
                    self.helper_true += 1
                }
                Some(Expr::Lit(lit)) if matches!(&lit.lit, syn::Lit::Bool(value) if !value.value) => {
                    self.helper_false += 1
                }
                _ => self.bad = true,
            }
            if self.function != "run_interleaved"
                || !self.guarded
                || self.depth != 0
                || !call.attrs.is_empty()
                || call.args.len() != 3
                || !state_arg
                || !prompt_arg
            {
                self.bad = true;
            }
            for arg in call.args.iter().skip(1) {
                self.visit_expr(arg);
            }
            return;
        }
        if path_is(&call.func, &["MetalQwen35State", "new"]) {
            self.constructors += 1;
            if !self.guarded
                || self.depth != 0
                || !matches!(self.function.as_str(), "run" | "run_interleaved")
            {
                self.bad = true;
            }
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let selected = matches!(
            call.method.to_string().as_str(),
            "reset_state" | "forward_prefill" | "set_gdn_chunked"
        );
        if path_is(&call.receiver, &["state"]) {
            if !self.guarded || self.depth != 0 || !call.attrs.is_empty() {
                self.bad = true;
            }
            match call.method.to_string().as_str() {
                "reset_state" if call.args.is_empty() => self.reset += 1,
                "forward_prefill" if call.args.len() == 1 => self.prefill += 1,
                "set_gdn_chunked" if call.args.len() == 1 => self.mode += 1,
                _ => self.bad = true,
            }
            for arg in &call.args {
                self.visit_expr(arg);
            }
            return;
        }
        if selected {
            self.bad = true;
        }
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.depth += 1;
        visit::visit_expr_closure(self, closure);
        self.depth -= 1;
    }
    fn visit_expr_async(&mut self, expr: &'ast syn::ExprAsync) {
        self.depth += 1;
        visit::visit_expr_async(self, expr);
        self.depth -= 1;
    }
    fn visit_item(&mut self, item: &'ast syn::Item) {
        self.depth += 1;
        visit::visit_item(self, item);
        self.depth -= 1;
    }
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        if attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr") {
            self.bad = true;
        }
        visit::visit_attribute(self, attr);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if sensitive(mac.tokens.clone()) {
            self.bad = true;
        }
        visit::visit_macro(self, mac);
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "parse")?;
    let find = |name: &str| {
        file.items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(function) if function.sig.ident == name => Some(function),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let (mains, runs, interleaved, helpers) = (
        find("main"),
        find("run"),
        find("run_interleaved"),
        find("run_one"),
    );
    let ([main], [run], [inter], [helper]) = (
        mains.as_slice(),
        runs.as_slice(),
        interleaved.as_slice(),
        helpers.as_slice(),
    ) else {
        return Err("functions");
    };
    if main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
        || !metal_cfg(&run.attrs)
        || !metal_cfg(&inter.attrs)
        || !metal_cfg(&helper.attrs)
        || run.sig.asyncness.is_some()
        || inter.sig.asyncness.is_some()
        || helper.sig.asyncness.is_some()
        || !helper.sig.inputs.first().is_some_and(state_param)
    {
        return Err("cfg or helper signature");
    }
    let entry = main.block.stmts.iter().filter(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs) && path_is(&call.func, &["run"]) && call.args.is_empty())).count();
    if entry != 1 {
        return Err("Metal main");
    }
    let positions = |function: &syn::ItemFn| {
        function
            .block
            .stmts
            .iter()
            .enumerate()
            .filter(|(_, stmt)| guard(stmt))
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    };
    let run_guards = positions(run);
    let inter_guards = positions(inter);
    let ([run_guard], [inter_guard]) = (run_guards.as_slice(), inter_guards.as_slice()) else {
        return Err("guards");
    };
    if run.block.stmts[..*run_guard]
        .iter()
        .filter(|stmt| guarded_interleaved_branch(stmt))
        .count()
        != 1
    {
        return Err("interleaved branch");
    }
    let mut all = Work::default();
    for item in &file.items {
        let Item::Fn(function) = item else { continue };
        all.function = function.sig.ident.to_string();
        for (index, stmt) in function.block.stmts.iter().enumerate() {
            if (all.function == "run" && index == *run_guard)
                || (all.function == "run_interleaved" && index == *inter_guard)
            {
                continue;
            }
            all.guarded = match all.function.as_str() {
                "run" => index > *run_guard,
                "run_interleaved" => index > *inter_guard,
                "run_one" => true,
                _ => false,
            };
            if all.function == "main"
                && let Stmt::Expr(Expr::Block(block), _) = stmt
            {
                if !nonmetal_cfg(&block.attrs) {
                    return Err("non-Metal main");
                }
                all.visit_block(&block.block);
                continue;
            }
            all.visit_stmt(stmt);
        }
    }
    if all.bad
        || all.main_calls != 1
        || all.interleaved_calls != 1
        || all.helper_calls != 6
        || all.helper_true != 3
        || all.helper_false != 3
        || all.state_bindings != 2
        || all.constructors != 2
        || all.reset != 3
        || all.prefill != 3
        || all.mode != 2
    {
        return Err("work chain");
    }
    Ok(())
}

#[test]
fn prefill_example_modes_keep_resets_and_prefills_guarded() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn prefill_example_contract_rejects_mutations() {
    let changes = [
        (
            "let _gpu_lock = lattice_inference::measurement::gpu_test_lock();",
            "let _gpu_lock = ();",
            "guards",
        ),
        (
            "let _gpu_lock = lattice_inference::measurement::gpu_test_lock();",
            "let _gpu_lock = lattice_inference::measurement::gpu_test_lock(); drop(_gpu_lock);",
            "work chain",
        ),
        (
            "        run_interleaved(&model_dir, &lens, runs, warmups, max_cache_len);",
            "        let later = || run_interleaved(&model_dir, &lens, runs, warmups, max_cache_len); later();",
            "interleaved branch",
        ),
        ("state.reset_state();", "other.reset_state();", "work chain"),
        (
            "state.reset_state();",
            "let later = || state.reset_state(); later();",
            "work chain",
        ),
        (
            "run_one(&mut state, &prompt_ids, false)",
            "run_one(&mut other, &prompt_ids, false)",
            "work chain",
        ),
        (
            "state.forward_prefill(prompt_ids)",
            "other.forward_prefill(prompt_ids)",
            "work chain",
        ),
        ("state.set_gdn_chunked(chunked);", "", "work chain"),
        (
            "run_one(&mut state, &prompt_ids, true)",
            "run_one(&mut state, &prompt_ids, false)",
            "work chain",
        ),
        ("    run();", "    other();", "Metal main"),
        (
            "        return;\n    }\n\n    let mode",
            "    }\n\n    let mode",
            "interleaved branch",
        ),
        (
            "state.reset_state();",
            "#[cfg(any())] state.reset_state();",
            "work chain",
        ),
    ];
    for (old, new, expected) in changes {
        assert!(SOURCE.contains(old), "missing mutation anchor: {old}");
        assert_eq!(
            check(&SOURCE.replacen(old, new, 1)),
            Err(expected),
            "mutation: {old}"
        );
    }
}
