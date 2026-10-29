//! Source-only guard contract for the optional Metal LoRA mixture benchmark.

use proc_macro2::{TokenStream, TokenTree};
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, Item, Meta, Pat, Stmt, Token};

const SOURCE: &str = include_str!("../src/bin/bench_lora_mixture.rs");

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
    let matches = |meta: &Meta, key: &str, expected: &str| match meta {
        Meta::NameValue(pair) if pair.path.is_ident(key) => matches!(
            &pair.value, Expr::Lit(expr) if matches!(&expr.lit, syn::Lit::Str(value) if value.value() == expected)
        ),
        _ => false,
    };
    args.len() == 2
        && matches(&args[0], "target_os", "macos")
        && matches(&args[1], "feature", "metal-gpu")
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

fn state_constructor(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Try(try_expr) = init.expr.as_ref() else {
        return false;
    };
    let Expr::MethodCall(map_err) = try_expr.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = map_err.receiver.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "metal"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && try_expr.attrs.is_empty()
        && map_err.attrs.is_empty()
        && map_err.method == "map_err"
        && map_err.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == 4
        && path_is(&call.func, &["MetalQwen35State", "from_q4_dir"])
}

fn sensitive(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "_gpu_lock"
                | "metal"
                | "gpu_test_lock"
                | "run_gpu_decode_bench"
                | "MetalQwen35State"
                | "load_lora_adapter"
                | "unload_lora_adapter"
                | "generate_with_lora_mixture"
        ),
        TokenTree::Group(group) => sensitive(group.stream()),
        TokenTree::Literal(literal) => syn::parse_str::<syn::LitStr>(&literal.to_string())
            .is_ok_and(|value| {
                value.value().split('{').skip(1).any(|part| {
                    let name: String = part
                        .chars()
                        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                        .collect();
                    name == "metal" || name == "_gpu_lock"
                })
            }),
        TokenTree::Punct(_) => false,
    })
}

#[derive(Default)]
struct Work {
    helper_identifiers: usize,
    helper_calls: usize,
    guards: usize,
    guard_uses: usize,
    lock_calls: usize,
    state_bindings: usize,
    state_paths: usize,
    constructors: usize,
    unload: usize,
    load: usize,
    generate: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}

impl<'ast> Visit<'ast> for Work {
    fn visit_ident(&mut self, id: &'ast syn::Ident) {
        self.helper_identifiers += usize::from(id == "run_gpu_decode_bench");
        visit::visit_ident(self, id);
    }
    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        self.attributes += 1;
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.guards += usize::from(pat.ident == "_gpu_lock");
        self.state_bindings += usize::from(pat.ident == "metal");
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.guard_uses += usize::from(path.path.is_ident("_gpu_lock"));
        self.state_paths += usize::from(path.path.is_ident("metal"));
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(&call.func, &["run_gpu_decode_bench"]) {
            self.helper_calls += 1;
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
        }
        if path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ) {
            self.lock_calls += 1;
        }
        if path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]) {
            self.constructors += 1;
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let selected = matches!(
            call.method.to_string().as_str(),
            "unload_lora_adapter" | "load_lora_adapter" | "generate_with_lora_mixture"
        );
        let state = path_is(&call.receiver, &["metal"]);
        if selected || state {
            self.bad |= !state || self.deferred > 0 || !call.attrs.is_empty();
            match (call.method.to_string().as_str(), call.args.len()) {
                ("unload_lora_adapter", 0) => self.unload += 1,
                ("load_lora_adapter", 3) => self.load += 1,
                ("generate_with_lora_mixture", 4) => self.generate += 1,
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

fn tuple_pattern(pat: &Pat, name: &str) -> bool {
    matches!(pat, Pat::TupleStruct(tuple) if tuple.path.is_ident(name) && tuple.elems.len() == 1)
}

fn metal_main_calls_run(function: &syn::ItemFn) -> bool {
    function.block.stmts.iter().any(|stmt| {
        let Stmt::Expr(Expr::Block(block), _) = stmt else {
            return false;
        };
        if !metal_cfg(&block.attrs) {
            return false;
        }
        let [Stmt::Expr(Expr::If(branch), _)] = block.block.stmts.as_slice() else {
            return false;
        };
        let Expr::Let(condition) = branch.cond.as_ref() else {
            return false;
        };
        let Expr::Call(call) = condition.expr.as_ref() else {
            return false;
        };
        tuple_pattern(&condition.pat, "Err")
            && path_is(&call.func, &["run"])
            && call.args.is_empty()
            && branch.else_branch.is_none()
    })
}

fn opt_in_gpu_call(stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::If(branch), _) = stmt else {
        return false;
    };
    let Expr::Let(condition) = branch.cond.as_ref() else {
        return false;
    };
    let Expr::Call(env) = condition.expr.as_ref() else {
        return false;
    };
    let Some(Expr::Lit(value)) = env.args.first() else {
        return false;
    };
    let [Stmt::Expr(Expr::Try(try_expr), Some(_))] = branch.then_branch.stmts.as_slice() else {
        return false;
    };
    let Expr::Call(call) = try_expr.expr.as_ref() else {
        return false;
    };
    tuple_pattern(&condition.pat, "Ok")
        && path_is(&env.func, &["std", "env", "var"])
        && env.args.len() == 1
        && matches!(&value.lit, syn::Lit::Str(text) if text.value() == "LATTICE_MODEL_DIR")
        && path_is(&call.func, &["run_gpu_decode_bench"])
        && call.args.len() == 5
        && branch.else_branch.is_some()
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source parse")?;
    let entry = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "main" => Some(function),
            _ => None,
        })
        .ok_or("Metal entry")?;
    if !metal_main_calls_run(entry) {
        return Err("Metal entry");
    }
    let run = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "run" => Some(function),
            _ => None,
        })
        .ok_or("optional GPU call")?;
    if !metal_cfg(&run.attrs)
        || run
            .block
            .stmts
            .iter()
            .filter(|stmt| opt_in_gpu_call(stmt))
            .count()
            != 1
    {
        return Err("optional GPU call");
    }
    let helpers: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "run_gpu_decode_bench" => Some(function),
            _ => None,
        })
        .collect();
    let [helper] = helpers.as_slice() else {
        return Err("Metal helper");
    };
    if helper.sig.asyncness.is_some() || !metal_cfg(&helper.attrs) {
        return Err("Metal helper");
    }
    let Some(guard_index) = helper.block.stmts.iter().position(guard) else {
        return Err("function guard");
    };
    let Some(state_index) = helper.block.stmts.iter().position(state_constructor) else {
        return Err("state constructor");
    };
    if guard_index >= state_index {
        return Err("guard after construction");
    }
    let mut local = Work::default();
    local.visit_block(&helper.block);
    if local.bad
        || local.attributes != 0
        || local.guards != 1
        || local.guard_uses != 0
        || local.lock_calls != 1
        || local.state_bindings != 1
        || local.state_paths != 7
        || local.constructors != 1
        || local.unload != 3
        || local.load != 2
        || local.generate != 2
    {
        return Err("guarded work changed");
    }
    let mut all = Work::default();
    for item in &file.items {
        if let Item::Fn(function) = item {
            all.visit_signature(&function.sig);
            all.visit_block(&function.block);
        } else {
            all.visit_item(item);
        }
    }
    if all.bad
        || all.helper_identifiers != 2
        || all.helper_calls != 1
        || all.constructors != 1
        || all.unload != 3
        || all.load != 2
        || all.generate != 2
    {
        return Err("unreviewed caller or Metal work");
    }
    Ok(())
}

#[test]
fn lora_mixture_metal_work_keeps_the_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn lora_mixture_contract_rejects_source_mutations() {
    let lock = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let constructor =
        "    let mut metal = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 512)";
    assert_eq!(SOURCE.matches(lock).count(), 1);
    assert_eq!(SOURCE.matches(constructor).count(), 1);
    let late_anchor = "    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)?;";
    assert_eq!(SOURCE.matches(late_anchor).count(), 1);
    let late = SOURCE
        .replacen(lock, "", 1)
        .replacen(late_anchor, &[lock, late_anchor].concat(), 1);
    assert!(syn::parse_file(&late).is_ok());
    assert_eq!(check(&late), Err("guard after construction"));
    for (before, after, expected) in [
        (
            "if let Err(e) = run()",
            "if let Err(e) = other()",
            "Metal entry",
        ),
        (
            "std::env::var(\"LATTICE_MODEL_DIR\")",
            "std::env::var(\"OTHER_MODEL_DIR\")",
            "optional GPU call",
        ),
        (lock, "", "function guard"),
        (
            lock,
            "    { let _gpu_lock = lattice_inference::measurement::gpu_test_lock(); }\n",
            "function guard",
        ),
        (
            lock,
            "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock(); drop(_gpu_lock);\n",
            "guarded work changed",
        ),
        (
            lock,
            "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock(); let _gpu_lock = ();\n",
            "guarded work changed",
        ),
        (
            constructor,
            "    let mut metal = OtherState::from_q4_dir(dir, &tokenizer_path, &cfg, 512)",
            "state constructor",
        ),
        (
            ".load_lora_adapter(blended, 1.0, None)",
            ".other_load(blended, 1.0, None)",
            "guarded work changed",
        ),
        (
            ".generate_with_lora_mixture(&refs, prompt, &tokenizer, &gen_cfg)",
            ".other_generate(&refs, prompt, &tokenizer, &gen_cfg)",
            "guarded work changed",
        ),
        (
            "    let cfg = Qwen35Config::qwen35_0_8b();",
            "    let _unguarded = MetalQwen35State::from_q4_dir(a,b,c,d); let cfg = Qwen35Config::qwen35_0_8b();",
            "unreviewed caller or Metal work",
        ),
        (
            "                metal.unload_lora_adapter();",
            "                other.unload_lora_adapter();",
            "guarded work changed",
        ),
        (
            "            metal.unload_lora_adapter();",
            "            let f = || metal.unload_lora_adapter();",
            "guarded work changed",
        ),
        (
            "            metal.unload_lora_adapter();",
            "            #[cfg(any())] metal.unload_lora_adapter();",
            "guarded work changed",
        ),
        (
            "            metal.unload_lora_adapter();",
            "            opaque!(metal); metal.unload_lora_adapter();",
            "guarded work changed",
        ),
        (
            "        run_gpu_decode_bench(&model_dir_str, new_tokens, warmup, iters, &ranks)?;",
            "        let _bypass = run_gpu_decode_bench; run_gpu_decode_bench(&model_dir_str, new_tokens, warmup, iters, &ranks)?;",
            "unreviewed caller or Metal work",
        ),
    ] {
        assert!(SOURCE.contains(before), "missing mutation anchor {before}");
        let mutant = SOURCE.replacen(before, after, 1);
        assert!(syn::parse_file(&mutant).is_ok(), "invalid mutant {before}");
        assert_eq!(check(&mutant), Err(expected), "mutation {before}");
    }
}
