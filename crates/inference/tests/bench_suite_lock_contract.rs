//! Source contracts for the two Metal-capable benchmark-suite arms.

#[path = "common/lock_contract.rs"]
mod lock_contract;

use lock_contract::path_is;
use syn::visit::Visit;
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_suite.rs");

fn metal_cfg(attrs: &[syn::Attribute]) -> bool {
    lock_contract::metal_cfg(attrs, false)
}

fn is_guard(stmt: &Stmt, conditional: bool) -> bool {
    lock_contract::guard(stmt, "_gpu_lock", conditional)
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
    lock_contract::sensitive_tokens(
        tokens,
        &[
            name,
            "_gpu_lock",
            "gpu_test_lock",
            "reset_state",
            "generate",
            "encode",
        ],
        &[name, "_gpu_lock"],
    )
}

struct SuitePolicy<'a> {
    name: &'a str,
    constructor: &'a [&'a str],
    bindings: usize,
    paths: usize,
    guards: usize,
    guard_calls: usize,
    constructors: usize,
    primary_calls: usize,
    reset_calls: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}
impl<'a> SuitePolicy<'a> {
    fn new(name: &'a str, constructor: &'a [&'a str]) -> Self {
        Self {
            name,
            constructor,
            bindings: 0,
            paths: 0,
            guards: 0,
            guard_calls: 0,
            constructors: 0,
            primary_calls: 0,
            reset_calls: 0,
            attributes: 0,
            deferred: 0,
            bad: false,
        }
    }
}
type Work<'a> = lock_contract::Work<SuitePolicy<'a>>;

impl lock_contract::WorkPolicy for SuitePolicy<'_> {
    fn deferred(&mut self) -> &mut usize {
        &mut self.deferred
    }
    fn attribute(&mut self, _attr: &syn::Attribute) {
        self.attributes += 1;
    }
    fn pat_ident(&mut self, pat: &syn::PatIdent) {
        self.bindings += usize::from(pat.ident == self.name);
        self.guards += usize::from(pat.ident == "_gpu_lock");
    }
    fn expr_path(&mut self, path: &syn::ExprPath) {
        self.paths += usize::from(path.path.is_ident(self.name));
        self.bad |= path.path.is_ident("_gpu_lock");
    }
    fn expr_call(&mut self, call: &syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.constructors += usize::from(path_is(&call.func, self.constructor));
    }
    fn expr_method_call(&mut self, call: &syn::ExprMethodCall) {
        if path_is(&call.receiver, &[self.name]) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            if (self.name == "metal_state" && call.method == "generate")
                || (self.name == "model" && call.method == "encode")
            {
                self.primary_calls += 1;
            } else if call.method == "reset_state" {
                self.reset_calls += 1;
            } else {
                self.bad = true;
            }
        } else if (self.name == "metal_state"
            && (call.method == "generate" || call.method == "reset_state"))
            || (self.name == "model" && call.method == "encode")
        {
            self.bad = true;
        }
    }
    fn macro_action(&mut self, mac: &syn::Macro) -> lock_contract::MacroAction {
        self.bad |= sensitive(mac.tokens.clone(), self.name);
        lock_contract::MacroAction::Walk
    }
}

fn check(source: &str, embedding: bool) -> Result<(), String> {
    let file = lock_contract::check(source)?;
    let (function, name, path, args, primary_calls, reset_calls) = if embedding {
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
        return Err("function changed".into());
    };
    let feature_cfg = |attrs: &[syn::Attribute]| {
        let [attr] = attrs else { return false };
        attr.path().is_ident("cfg")
            && matches!(attr.parse_args::<Meta>(), Ok(Meta::NameValue(nv))
                if nv.path.is_ident("feature") && matches!(&nv.value,
                    Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == "metal-gpu")))
    };
    if function.sig.asyncness.is_some()
        || (embedding && !function.attrs.is_empty())
        || (!embedding && !feature_cfg(&function.attrs))
    {
        return Err("function cfg or execution changed".into());
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
        return Err("guard or constructor changed".into());
    };
    if guard >= binding {
        return Err("guard must precede constructor".into());
    }
    let mut all = Work::new(SuitePolicy::new(name, path));
    all.visit_block(&function.block);
    let mut before = Work::new(SuitePolicy::new(name, path));
    for stmt in &function.block.stmts[..*binding] {
        before.visit_stmt(stmt);
    }
    if all.bad {
        return Err("unreviewed guard or work use".into());
    }
    lock_contract::check_counts(&[
        ("attributes", all.attributes, usize::from(embedding)),
        ("guards", all.guards, 1),
        ("guard_calls", all.guard_calls, 1),
        ("bindings", all.bindings, 1),
        ("paths", all.paths, primary_calls + reset_calls),
        ("primary_calls", all.primary_calls, primary_calls),
        ("reset_calls", all.reset_calls, reset_calls),
        ("constructors", all.constructors, 1),
        ("before.paths", before.paths, 0),
        ("before.primary_calls", before.primary_calls, 0),
        ("before.reset_calls", before.reset_calls, 0),
    ])?;
    Ok(())
}

fn hidden_gpu_work_mutant(source: &str, entry: &str, guard: &str) -> String {
    const DEFINITION: &str = "macro_rules! hidden_gpu_work { () => {{ let _hidden_guard = lattice_inference::measurement::gpu_test_lock(); }}; }\n";
    let entry_index = source.find(entry).expect("missing entry mutation anchor");
    let cfg_index = source[..entry_index]
        .rfind("#[cfg(")
        .expect("missing entry cfg mutation anchor");
    assert!(source.contains(guard), "missing guard mutation anchor");
    let mut mutated = source.to_owned();
    mutated.insert_str(cfg_index, DEFINITION);
    let mutated = mutated.replacen(guard, &format!("{guard}\n    hidden_gpu_work!();\n"), 1);
    assert!(
        syn::parse_file(&mutated).is_ok(),
        "hidden macro mutant must parse"
    );
    mutated
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
    for (name, anchor) in [
        ("guard", guard),
        ("conditional cfg", cfg),
        ("metal work", metal_work),
        ("embedding work", embed_work),
    ] {
        assert!(SOURCE.contains(anchor), "missing {name} mutation anchor");
    }
    let metal_guard = format!("{guard}    let t_init = Instant::now();");
    let embed_guard = format!("{cfg}{guard}    let model =");
    assert!(
        SOURCE.contains(metal_guard.as_str()),
        "missing metal guard anchor"
    );
    assert!(
        SOURCE.contains(embed_guard.as_str()),
        "missing embedding guard anchor"
    );
    assert!(
        SOURCE.contains("#[cfg(feature = \"metal-gpu\")]\nfn bench_llm_metal()"),
        "missing function cfg anchor"
    );
    assert!(
        SOURCE.contains("metal_state.generate(\"Hello\""),
        "missing generation anchor"
    );
    let metal_mutations = [
        hidden_gpu_work_mutant(SOURCE, "fn bench_llm_metal(", metal_guard.as_str()),
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
    ];
    let metal_cases = [
        ("hidden macro GPU work", "opaque or shadowed macro"),
        ("function cfg", "function cfg or execution changed"),
        ("removed guard", "guard or constructor changed"),
        ("dropped guard", "unreviewed guard or work use"),
        ("shadowed guard", "guards: observed 2, expected 1"),
        ("other reset receiver", "unreviewed guard or work use"),
        ("changed generation method", "unreviewed guard or work use"),
        ("conditional reset", "unreviewed guard or work use"),
        ("deferred reset", "unreviewed guard or work use"),
        ("removed reset", "paths: observed 3, expected 4"),
        ("duplicated reset", "paths: observed 5, expected 4"),
        ("late guard", "guard must precede constructor"),
        ("captured state", "unreviewed guard or work use"),
    ];
    assert_eq!(metal_cases.len(), metal_mutations.len());
    for ((name, expected), mutated) in metal_cases.into_iter().zip(metal_mutations) {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        assert_eq!(check(&mutated, false), Err(expected.into()), "{name}");
    }
    let embedding_mutations = [
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
    ];
    let embedding_cases = [
        ("removed guard", "guard or constructor changed"),
        ("dropped guard", "unreviewed guard or work use"),
        ("shadowed guard", "guards: observed 2, expected 1"),
        ("weakened guard cfg", "guard or constructor changed"),
        ("other encode receiver", "unreviewed guard or work use"),
        ("conditional encode", "attributes: observed 2, expected 1"),
        ("deferred encode", "unreviewed guard or work use"),
        ("removed encode", "paths: observed 1, expected 2"),
        ("duplicated encode", "paths: observed 3, expected 2"),
        ("late guard", "guard must precede constructor"),
        ("captured model", "unreviewed guard or work use"),
    ];
    assert_eq!(embedding_cases.len(), embedding_mutations.len());
    for ((name, expected), mutated) in embedding_cases.into_iter().zip(embedding_mutations) {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        assert_eq!(check(&mutated, true), Err(expected.into()), "{name}");
    }
}
