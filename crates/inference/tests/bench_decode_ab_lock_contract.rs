//! Source contract for A/B Metal chat completion and reset work.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../src/bin/bench_decode_ab.rs");

fn path_is(expr: &Expr, segments: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == segments.len()
        && path.path.segments.iter().zip(segments).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
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

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "metal"
                | "_gpu_lock"
                | "gpu_test_lock"
                | "chat_completion"
                | "reset_state"
                | "run"
                | "MetalQwen35State"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let name: String = part
                        .chars()
                        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                        .collect();
                    name == "metal" || name == "_gpu_lock"
                })
            })
        }
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    chat_completion: usize,
    constructors: usize,
    state_imports: usize,
    invalid_state_binding: bool,
    assignments: usize,
    q4_assignments: usize,
    f16_assignments: usize,
    reset: usize,
    receiver_paths: usize,
    metal_bindings: usize,
    guard_bindings: usize,
    guard_calls: usize,
    run_calls: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}

impl<'ast> Visit<'ast> for Work {
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        fn mentions_state(tree: &syn::UseTree) -> bool {
            match tree {
                syn::UseTree::Path(path) => {
                    path.ident == "MetalQwen35State" || mentions_state(&path.tree)
                }
                syn::UseTree::Name(name) => name.ident == "MetalQwen35State",
                syn::UseTree::Rename(rename) => {
                    rename.ident == "MetalQwen35State" || rename.rename == "MetalQwen35State"
                }
                syn::UseTree::Group(group) => group.items.iter().any(mentions_state),
                syn::UseTree::Glob(_) => false,
            }
        }
        if mentions_state(&item.tree) {
            self.state_imports += 1;
            let valid = (|| {
                let syn::UseTree::Path(root) = &item.tree else {
                    return false;
                };
                if root.ident != "lattice_inference" && root.ident != "lattice_model_qwen35" {
                    return false;
                }
                let syn::UseTree::Path(forward) = root.tree.as_ref() else {
                    return false;
                };
                if forward.ident != "forward" {
                    return false;
                }
                let syn::UseTree::Path(metal) = forward.tree.as_ref() else {
                    return false;
                };
                if metal.ident != "metal_qwen35" {
                    return false;
                }
                let syn::UseTree::Group(group) = metal.tree.as_ref() else {
                    return false;
                };
                let mut names = std::collections::BTreeSet::new();
                for leaf in &group.items {
                    let syn::UseTree::Name(name) = leaf else {
                        return false;
                    };
                    if !names.insert(name.ident.to_string()) {
                        return false;
                    }
                }
                names
                    == std::collections::BTreeSet::from([
                        "ChatMessage".to_string(),
                        "MetalQwen35State".to_string(),
                    ])
            })();
            self.invalid_state_binding |=
                !valid || !item.attrs.is_empty() || item.leading_colon.is_some();
        }
        visit::visit_item_use(self, item);
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.invalid_state_binding |= item.ident == "MetalQwen35State";
        visit::visit_item_struct(self, item);
    }
    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        self.invalid_state_binding |= item.ident == "MetalQwen35State";
        visit::visit_item_enum(self, item);
    }
    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.invalid_state_binding |= item.ident == "MetalQwen35State";
        visit::visit_item_type(self, item);
    }
    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        if path_is(&assign.left, &["metal"]) {
            self.assignments += 1;
            self.bad |= self.deferred > 0 || !assign.attrs.is_empty();
            let (method, receiver, outer_attrs_ok) = match assign.right.as_ref() {
                Expr::Try(try_expr) => match try_expr.expr.as_ref() {
                    Expr::MethodCall(map) => (
                        map,
                        map.receiver.as_ref(),
                        try_expr.attrs.is_empty() && map.method == "map_err",
                    ),
                    _ => {
                        self.bad = true;
                        return;
                    }
                },
                Expr::MethodCall(expect) => {
                    (expect, expect.receiver.as_ref(), expect.method == "expect")
                }
                _ => {
                    self.bad = true;
                    return;
                }
            };
            let Expr::Call(call) = receiver else {
                self.bad = true;
                return;
            };
            self.bad |= !outer_attrs_ok
                || !method.attrs.is_empty()
                || method.args.len() != 1
                || !call.attrs.is_empty();
            if path_is(&call.func, &["MetalQwen35State", "from_q4_dir"])
                && call.args.len() == 4
                && method.method == "map_err"
            {
                self.q4_assignments += 1;
            } else if path_is(&call.func, &["MetalQwen35State", "new"])
                && call.args.len() == 3
                && method.method == "expect"
            {
                self.f16_assignments += 1;
            } else {
                self.bad = true;
            }
        }
        visit::visit_expr_assign(self, assign);
    }
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attributes += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.invalid_state_binding |= pat.ident == "MetalQwen35State";
        self.metal_bindings += usize::from(pat.ident == "metal");
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.is_ident("metal") {
            self.receiver_paths += 1;
        }
        if path.path.is_ident("_gpu_lock") {
            self.bad = true;
        }
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(&call.func, &["MetalQwen35State", "from_q4_dir"])
            || path_is(&call.func, &["MetalQwen35State", "new"])
        {
            self.constructors += 1;
        }
        if path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ) {
            self.guard_calls += 1;
        }
        if path_is(&call.func, &["run"]) {
            self.run_calls += 1;
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "chat_completion" || call.method == "reset_state" {
            self.bad |=
                self.deferred > 0 || !call.attrs.is_empty() || !path_is(&call.receiver, &["metal"]);
            self.chat_completion += usize::from(call.method == "chat_completion");
            self.reset += usize::from(call.method == "reset_state");
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

fn direct_entry_calls_run(block: &syn::Block) -> bool {
    let [Stmt::Expr(Expr::If(branch), None)] = block.stmts.as_slice() else {
        return false;
    };
    let Expr::Let(condition) = branch.cond.as_ref() else {
        return false;
    };
    let Pat::TupleStruct(error) = condition.pat.as_ref() else {
        return false;
    };
    let Expr::Call(call) = condition.expr.as_ref() else {
        return false;
    };
    branch.attrs.is_empty()
        && branch.else_branch.is_none()
        && condition.attrs.is_empty()
        && error.path.is_ident("Err")
        && error.elems.len() == 1
        && call.attrs.is_empty()
        && call.args.is_empty()
        && path_is(&call.func, &["run"])
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let find_fn = |name: &str| {
        file.items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(f) if f.sig.ident == name => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let runs = find_fn("run");
    let mains = find_fn("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main");
    };
    if !expected_metal_cfg(&run.attrs) || run.sig.asyncness.is_some() {
        return Err("run cfg changed");
    }
    let guards: Vec<_> = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect();
    let [(guard_index, _)] = guards.as_slice() else {
        return Err("expected one direct guard");
    };
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
    let mut entry = Work::default();
    entry.visit_block(&main.block);
    let entry_blocks = main
        .block
        .stmts
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Expr(Expr::Block(block), _) if expected_metal_cfg(&block.attrs) => Some(block),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [entry_block] = entry_blocks.as_slice() else {
        return Err("expected Metal main entry block");
    };
    let mut guarded_entry = Work::default();
    guarded_entry.visit_block(&entry_block.block);
    if main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
        || guarded_entry.run_calls != 1
        || !direct_entry_calls_run(&entry_block.block)
    {
        return Err("Metal main entry changed");
    }
    let declarations = run.block.stmts.iter().enumerate().filter(|(_, stmt)| match stmt {
        Stmt::Local(local) => match &local.pat {
            Pat::Type(typed) => matches!(typed.pat.as_ref(), Pat::Ident(binding)
                if binding.ident == "metal" && binding.mutability.is_some()
                    && binding.by_ref.is_none() && binding.subpat.is_none())
                && matches!(typed.ty.as_ref(), syn::Type::Path(ty) if ty.qself.is_none() && ty.path.is_ident("MetalQwen35State"))
                && local.attrs.is_empty() && typed.attrs.is_empty() && local.init.is_none(),
            _ => false,
        },
        _ => false,
    }).collect::<Vec<_>>();
    let [(metal_index, _)] = declarations.as_slice() else {
        return Err("expected typed Metal binding");
    };
    if guard_index >= metal_index {
        return Err("guard must precede state binding");
    }
    let branches = run
        .block
        .stmts
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Expr(Expr::If(branch), _) if path_is(&branch.cond, &["is_q4"]) => Some(branch),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [branch] = branches.as_slice() else {
        return Err("expected Q4/f16 branch");
    };
    let Some((_, otherwise)) = &branch.else_branch else {
        return Err("missing f16 branch");
    };
    let Expr::Block(otherwise) = otherwise.as_ref() else {
        return Err("unexpected f16 branch");
    };
    let mut q4 = Work::default();
    q4.visit_block(&branch.then_branch);
    let mut f16 = Work::default();
    f16.visit_block(&otherwise.block);
    if q4.bad
        || f16.bad
        || q4.assignments != 1
        || q4.q4_assignments != 1
        || q4.f16_assignments != 0
        || f16.assignments != 1
        || f16.q4_assignments != 0
        || f16.f16_assignments != 1
    {
        return Err("unreviewed Q4/f16 construction");
    }
    if all.bad
        || inside.bad
        || inside.attributes != 0
        || all.guard_calls != 1
        || all.guard_bindings != 1
        || inside.guard_bindings != 1
        || all.metal_bindings != 1
        || inside.metal_bindings != 1
        || all.receiver_paths != 6
        || inside.receiver_paths != 6
        || all.chat_completion != 2
        || inside.chat_completion != 2
        || all.reset != 2
        || inside.reset != 2
        || all.assignments != 2
        || inside.assignments != 2
        || all.invalid_state_binding
        || inside.invalid_state_binding
        || all.state_imports != 1
        || inside.state_imports != 1
        || all.constructors != 2
        || inside.constructors != 2
        || before.constructors != 0
        || all.q4_assignments != 1
        || inside.q4_assignments != 1
        || all.f16_assignments != 1
        || inside.f16_assignments != 1
        || before.assignments != 0
        || before.chat_completion != 0
        || before.reset != 0
        || before.metal_bindings != 0
        || all.run_calls != 1
        || entry.run_calls != 1
    {
        return Err("unreviewed guard, state, work, or entrypoint use");
    }
    Ok(())
}

const FACADE_IMPORTS: &str = r#"    use lattice_inference::GenerateConfig;
    use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::model_format::{ModelFormat, detect_format};
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};"#;
const OWNER_IMPORTS: &str = r#"    use lattice_arch::generation::GenerateConfig;
    use lattice_arch::qwen35_config::Qwen35Config;
    use lattice_model_qwen35::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
    use lattice_model_qwen35::model::qwen35::Qwen35Model;
    use lattice_tokenizer::{BpeTokenizer, Tokenizer};
    use lattice_weights::model_format::{ModelFormat, detect_format};"#;

fn replace_once(source: &str, before: &str, after: &str) -> String {
    assert_eq!(
        source.matches(before).count(),
        1,
        "mutation anchor {before}"
    );
    source.replacen(before, after, 1)
}

fn current_imports() -> (&'static str, &'static str) {
    match (
        SOURCE.matches(FACADE_IMPORTS).count(),
        SOURCE.matches(OWNER_IMPORTS).count(),
    ) {
        (1, 0) => (FACADE_IMPORTS, OWNER_IMPORTS),
        (0, 1) => (OWNER_IMPORTS, FACADE_IMPORTS),
        _ => panic!("unexpected target imports"),
    }
}

#[test]
fn decode_ab_chat_and_resets_keep_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
    let (current, alternative) = current_imports();
    assert_eq!(check(&replace_once(SOURCE, current, alternative)), Ok(()));
}

fn reject(mutant: String, expected: &str) {
    assert!(syn::parse_file(&mutant).is_ok(), "mutation must parse");
    assert_eq!(check(&mutant), Err(expected));
}

#[test]
fn decode_ab_contract_rejects_real_source_mutations() {
    const GENERIC: &str = "unreviewed guard, state, work, or entrypoint use";
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let declaration = "    let mut metal: MetalQwen35State;\n";
    let reset = "\n    metal.reset_state();\n";
    let chat = "        let result = metal.chat_completion(&history, &tokenizer, &gen_cfg)?;";
    let entry = "if let Err(e) = run()";
    let changed = |before: &str, after: &str| replace_once(SOURCE, before, after);

    reject(changed(guard, ""), "expected one direct guard");
    reject(
        changed(guard, &format!("    {{ {guard}    }}\n")),
        "expected one direct guard",
    );
    let late = changed(guard, "");
    reject(
        replace_once(&late, declaration, &format!("{declaration}{guard}")),
        "guard must precede state binding",
    );
    reject(
        changed(guard, &format!("{guard}    drop(_gpu_lock);\n")),
        GENERIC,
    );
    reject(
        changed(guard, &format!("{guard}    let _gpu_lock = ();\n")),
        GENERIC,
    );
    reject(changed(reset, "\n"), GENERIC);
    reject(changed(reset, &format!("{reset}{reset}")), GENERIC);
    reject(
        changed(reset, "\n    let _deferred = || metal.reset_state();\n"),
        GENERIC,
    );
    reject(
        changed(reset, "\n    #[cfg(any())]\n    metal.reset_state();\n"),
        GENERIC,
    );
    reject(
        changed(
            chat,
            "        let result = other.chat_completion(&history, &tokenizer, &gen_cfg)?;",
        ),
        GENERIC,
    );
    reject(
        changed(
            guard,
            &format!("{guard}    println!(\"{{_gpu_lock:?}}\");\n"),
        ),
        GENERIC,
    );
    reject(
        changed(chat, &format!("{chat}\n        println!(\"{{metal:?}}\");")),
        GENERIC,
    );
    reject(
        changed("MetalQwen35State::from_q4_dir(", "OtherState::from_q4_dir("),
        "unreviewed Q4/f16 construction",
    );
    reject(
        changed(
            declaration,
            &format!("{declaration}    let _extra = MetalQwen35State::new(a, b, c);\n"),
        ),
        GENERIC,
    );
    let source_import =
        "use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};";
    let owner_import =
        "use lattice_model_qwen35::forward::metal_qwen35::{ChatMessage, MetalQwen35State};";
    let actual_import = if SOURCE.contains(source_import) {
        source_import
    } else {
        owner_import
    };
    reject(
        changed(
            actual_import,
            "use fake::forward::metal_qwen35::{ChatMessage, MetalQwen35State};",
        ),
        GENERIC,
    );
    reject(
        changed(
            entry,
            "if let Err(e) = (if false { run() } else { Ok(()) })",
        ),
        "Metal main entry changed",
    );
    reject(
        changed(entry, "if let Err(e) = (|| run())()"),
        "Metal main entry changed",
    );
}
