//! Source contract for reset work in two Metal GDN profiling examples.

#[path = "support/lock_contract.rs"]
mod lock_contract;

use lock_contract::path_is;
use syn::visit::{self, Visit};
use syn::{Expr, Item, Pat, Stmt};

const DECODE: &str = include_str!("../examples/decode_profile.rs");
const GDN: &str = include_str!("../examples/bench_gdn_state.rs");
const ALLOCATION: &str = include_str!("../examples/bench_metal_forward_allocation.rs");

fn metal_cfg(attrs: &[syn::Attribute], counters: bool) -> bool {
    lock_contract::metal_cfg(attrs, counters)
}

fn direct_guard(stmt: &Stmt) -> bool {
    lock_contract::guard(stmt, "_gpu_lock", false)
}

#[derive(Default)]
struct Constructors {
    q4: usize,
    f16: usize,
}
impl<'ast> Visit<'ast> for Constructors {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]) {
            self.q4 += 1;
        }
        if path_is(&call.func, &["MetalQwen35State", "new"]) {
            self.f16 += 1;
        }
        visit::visit_expr_call(self, call);
    }
}

fn state_binding(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Type(typed) = &local.pat else {
        return false;
    };
    let Pat::Ident(binding) = typed.pat.as_ref() else {
        return false;
    };
    let syn::Type::Path(ty) = typed.ty.as_ref() else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::If(branch) = init.expr.as_ref() else {
        return false;
    };
    let Some((_, otherwise)) = &branch.else_branch else {
        return false;
    };
    let Expr::Block(otherwise) = otherwise.as_ref() else {
        return false;
    };
    if !local.attrs.is_empty()
        || !typed.attrs.is_empty()
        || !branch.attrs.is_empty()
        || !otherwise.attrs.is_empty()
        || binding.ident != "state"
        || binding.mutability.is_none()
        || binding.by_ref.is_some()
        || binding.subpat.is_some()
        || ty.qself.is_some()
        || !ty.path.is_ident("MetalQwen35State")
        || init.diverge.is_some()
        || !path_is(&branch.cond, &["is_q4"])
    {
        return false;
    }
    let mut q4 = Constructors::default();
    q4.visit_block(&branch.then_branch);
    let mut f16 = Constructors::default();
    f16.visit_block(&otherwise.block);
    q4.q4 == 1 && q4.f16 == 0 && f16.q4 == 0 && f16.f16 == 1
}

fn captures_sensitive_format(s: &str) -> bool {
    lock_contract::captures_format(s, &["state", "_gpu_lock"])
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    lock_contract::sensitive_tokens(
        tokens,
        &[
            "state",
            "_gpu_lock",
            "gpu_test_lock",
            "reset_state",
            "forward_prefill",
            "forward_step",
        ],
        &["state", "_gpu_lock"],
    )
}

#[derive(Default)]
struct ProfilerPolicy {
    state_bindings: usize,
    guard_bindings: usize,
    guard_calls: usize,
    run_calls: usize,
    state_paths: usize,
    reset: usize,
    prefill: usize,
    step: usize,
    report: usize,
    take_report: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}
type Work = lock_contract::Work<ProfilerPolicy>;

impl lock_contract::WorkPolicy for ProfilerPolicy {
    fn deferred(&mut self) -> &mut usize {
        &mut self.deferred
    }
    fn attribute(&mut self, _attr: &syn::Attribute) {
        self.attributes += 1;
    }
    fn pat_ident(&mut self, pat: &syn::PatIdent) {
        self.state_bindings += usize::from(pat.ident == "state");
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
    }
    fn expr_path(&mut self, path: &syn::ExprPath) {
        if path.path.is_ident("state") {
            self.state_paths += 1;
        }
        if path.path.is_ident("_gpu_lock") {
            self.bad = true;
        }
    }
    fn expr_call(&mut self, call: &syn::ExprCall) {
        if path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ) {
            self.guard_calls += 1;
        }
        if path_is(&call.func, &["run"]) {
            self.run_calls += 1;
        }
    }
    fn expr_method_call(&mut self, call: &syn::ExprMethodCall) {
        let direct = path_is(&call.receiver, &["state"]);
        let work_method = call.method == "reset_state"
            || call.method == "forward_prefill"
            || call.method == "forward_step";
        if direct {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            match call.method.to_string().as_str() {
                "reset_state" => self.reset += 1,
                "forward_prefill" => self.prefill += 1,
                "forward_step" => self.step += 1,
                "gdn_state_traffic_report" => self.report += 1,
                "take_gdn_state_traffic_report" => self.take_report += 1,
                _ => self.bad = true,
            }
        } else if work_method {
            self.bad = true;
        }
    }
    fn macro_action(&mut self, mac: &syn::Macro) -> lock_contract::MacroAction {
        self.bad |= sensitive(mac.tokens.clone());
        lock_contract::MacroAction::Walk
    }
}

fn check(source: &str, counters: bool) -> Result<(), String> {
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
    let runs = find("run");
    let mains = find("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main".into());
    };
    if !metal_cfg(&run.attrs, counters)
        || run.sig.asyncness.is_some()
        || main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
    {
        return Err("entry cfg changed".into());
    }
    let entry = main.block.stmts.iter().filter(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs, counters) && path_is(&call.func, &["run"]) && call.args.is_empty())).count();
    if entry != 1 {
        return Err("expected direct Metal entry call".into());
    }
    let guards = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect::<Vec<_>>();
    let states = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| state_binding(stmt))
        .collect::<Vec<_>>();
    let ([(g, _)], [(s, _)]) = (guards.as_slice(), states.as_slice()) else {
        return Err("guard or typed state missing".into());
    };
    if g >= s {
        return Err("guard must precede state".into());
    }
    let all = lock_contract::visit_file_bodies(&file, Work::default());
    let mut inside = Work::default();
    inside.visit_block(&run.block);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*g] {
        before.visit_stmt(stmt);
    }
    let reports = usize::from(counters);
    let takes = if counters { 2 } else { 0 };
    let paths = 4 + reports + takes;
    if all.bad || inside.bad {
        return Err("unreviewed guard, state or work use".into());
    }
    lock_contract::check_counts(&[
        ("inside.attributes", inside.attributes, 0),
        ("all.guard_calls", all.guard_calls, 1),
        ("all.guard_bindings", all.guard_bindings, 1),
        ("inside.guard_bindings", inside.guard_bindings, 1),
        ("all.state_bindings", all.state_bindings, 1),
        ("inside.state_bindings", inside.state_bindings, 1),
        ("all.state_paths", all.state_paths, paths),
        ("inside.state_paths", inside.state_paths, paths),
        ("all.reset", all.reset, 1),
        ("inside.reset", inside.reset, 1),
        ("all.prefill", all.prefill, 1),
        ("inside.prefill", inside.prefill, 1),
        ("all.step", all.step, 2),
        ("inside.step", inside.step, 2),
        ("all.report", all.report, reports),
        ("inside.report", inside.report, reports),
        ("all.take_report", all.take_report, takes),
        ("inside.take_report", inside.take_report, takes),
        ("before.state_bindings", before.state_bindings, 0),
        ("before.reset", before.reset, 0),
        ("before.prefill", before.prefill, 0),
        ("before.step", before.step, 0),
        ("all.run_calls", all.run_calls, 1),
    ])?;
    Ok(())
}

#[test]
fn profilers_keep_resets_and_forward_work_under_guard() {
    assert_eq!(check(DECODE, false), Ok(()));
    assert_eq!(check(GDN, true), Ok(()));
}

#[test]
fn profiler_contract_rejects_real_source_mutations() {
    for (source, counters) in [(DECODE, false), (GDN, true)] {
        let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
        let reset = "    state.reset_state();";
        let prefill = "state.forward_prefill(&prompt_ids)";
        for (name, anchor) in [
            ("guard", guard),
            ("reset", reset),
            ("prefill", prefill),
            ("main", "fn main() {"),
        ] {
            assert!(source.contains(anchor), "missing {name} mutation anchor");
        }

        let paths = 4 + usize::from(counters) + if counters { 2 } else { 0 };
        let mutations = [
            (
                "hidden macro GPU work",
                lock_contract::hidden_gpu_work_mutant(source, "fn run(", guard),
                "opaque or shadowed macro".to_owned(),
            ),
            (
                "removed guard",
                source.replacen(guard, "", 1),
                "guard or typed state missing".to_owned(),
            ),
            (
                "conditional main",
                source.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
                "entry cfg changed".to_owned(),
            ),
            (
                "dropped guard",
                source.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
                "unreviewed guard, state or work use".to_owned(),
            ),
            (
                "shadowed guard",
                source.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
                "all.guard_bindings: observed 2, expected 1".to_owned(),
            ),
            (
                "other reset receiver",
                source.replacen(reset, "    other.reset_state();", 1),
                "unreviewed guard, state or work use".to_owned(),
            ),
            (
                "removed reset",
                source.replacen(reset, "", 1),
                format!("all.state_paths: observed {}, expected {paths}", paths - 1),
            ),
            (
                "duplicated reset",
                source.replacen(reset, &format!("{reset}\n{reset}"), 1),
                format!("all.state_paths: observed {}, expected {paths}", paths + 1),
            ),
            (
                "deferred reset",
                source.replacen(reset, "    let deferred = || state.reset_state();", 1),
                "unreviewed guard, state or work use".to_owned(),
            ),
            (
                "conditional reset",
                source.replacen(reset, &format!("    #[cfg(any())]\n{reset}"), 1),
                "unreviewed guard, state or work use".to_owned(),
            ),
            (
                "macro guard use",
                source.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
                "opaque or shadowed macro".to_owned(),
            ),
            (
                "captured state",
                source.replacen(
                    reset,
                    &format!("{reset}\n    println!(\"{{state:?}}\");"),
                    1,
                ),
                "unreviewed guard, state or work use".to_owned(),
            ),
            (
                "other prefill receiver",
                source.replacen(prefill, "other.forward_prefill(&prompt_ids)", 1),
                "unreviewed guard, state or work use".to_owned(),
            ),
            (
                "late guard",
                source
                    .replacen(guard, "", 1)
                    .replacen(reset, &format!("{reset}\n{guard}"), 1),
                "guard must precede state".to_owned(),
            ),
        ];

        for (name, mutated, expected) in mutations {
            assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
            assert_eq!(check(&mutated, counters), Err(expected), "{name}");
        }
    }
}

#[test]
fn documented_main_and_both_metal_cfg_forms_are_recognized() {
    for (source, counters) in [(DECODE, false), (GDN, true)] {
        let file = syn::parse_file(source).unwrap();
        let main = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(f) if f.sig.ident == "main" => Some(f),
                _ => None,
            })
            .unwrap();
        assert!(!main.attrs.is_empty());
        assert!(main.attrs.iter().all(|attr| attr.path().is_ident("doc")));
        assert!(main.block.stmts.iter().any(|stmt| matches!(stmt,
            Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs, counters) && path_is(&call.func, &["run"]))));
    }
    assert!(!captures_sensitive_format("{state_bytes_moved}"));
    assert!(captures_sensitive_format("{state:?}"));
}

fn allocation_state_binding(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Match(branch) = init.expr.as_ref() else {
        return false;
    };
    let Expr::Call(call) = branch.expr.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && branch.attrs.is_empty()
        && call.attrs.is_empty()
        && call.args.len() == 4
        && path_is(&call.func, &["MetalQwen35State", "from_q4_dir"])
}

fn check_allocation(source: &str) -> Result<(), String> {
    let file = lock_contract::check(source)?;
    let find = |name: &str| {
        file.items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(function) if function.sig.ident == name => Some(function),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let runs = find("run");
    let mains = find("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main".into());
    };
    if !metal_cfg(&run.attrs, false)
        || run.sig.asyncness.is_some()
        || main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
    {
        return Err("entry cfg changed".into());
    }
    let entry = main
        .block
        .stmts
        .iter()
        .filter(|stmt| {
            matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs, false)
            && path_is(&call.func, &["run"]) && call.args.is_empty())
        })
        .count();
    if entry != 1 {
        return Err("expected direct Metal entry".into());
    }
    let guards: Vec<_> = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect();
    let states: Vec<_> = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| allocation_state_binding(stmt))
        .collect();
    let ([(g, _)], [(s, _)]) = (guards.as_slice(), states.as_slice()) else {
        return Err("guard or allocation state missing".into());
    };
    if g >= s {
        return Err("guard must precede state".into());
    }
    let all = lock_contract::visit_file_bodies(&file, Work::default());
    let mut inside = Work::default();
    inside.visit_block(&run.block);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*g] {
        before.visit_stmt(stmt);
    }
    let mut constructors = Constructors::default();
    constructors.visit_file(&file);
    if all.bad || inside.bad {
        return Err("unreviewed guard, state or allocation work".into());
    }
    lock_contract::check_counts(&[
        ("inside.attributes", inside.attributes, 0),
        ("all.guard_calls", all.guard_calls, 1),
        ("all.guard_bindings", all.guard_bindings, 1),
        ("inside.guard_bindings", inside.guard_bindings, 1),
        ("all.state_bindings", all.state_bindings, 1),
        ("inside.state_bindings", inside.state_bindings, 1),
        ("all.state_paths", all.state_paths, 3),
        ("inside.state_paths", inside.state_paths, 3),
        ("all.reset", all.reset, 1),
        ("inside.reset", inside.reset, 1),
        ("all.prefill", all.prefill, 0),
        ("inside.prefill", inside.prefill, 0),
        ("all.step", all.step, 2),
        ("inside.step", inside.step, 2),
        ("all.run_calls", all.run_calls, 1),
        ("before.state_bindings", before.state_bindings, 0),
        ("before.reset", before.reset, 0),
        ("before.step", before.step, 0),
        ("constructors.q4", constructors.q4, 1),
        ("constructors.f16", constructors.f16, 0),
    ])?;
    Ok(())
}

#[test]
fn allocation_profiler_keeps_reset_and_forward_work_under_guard() {
    assert_eq!(check_allocation(ALLOCATION), Ok(()));
}

#[test]
fn allocation_profiler_contract_rejects_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "    state.reset_state();";
    for (name, anchor) in [
        ("guard", guard),
        ("reset", reset),
        ("step", "state.forward_step(0, 1)"),
        ("entry", "    run();"),
    ] {
        assert!(
            ALLOCATION.contains(anchor),
            "missing {name} mutation anchor"
        );
    }

    let mutations = [
        (
            "hidden macro GPU work",
            lock_contract::hidden_gpu_work_mutant(ALLOCATION, "fn run(", guard),
            "opaque or shadowed macro",
        ),
        (
            "removed guard",
            ALLOCATION.replacen(guard, "", 1),
            "guard or allocation state missing",
        ),
        (
            "dropped guard",
            ALLOCATION.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
            "unreviewed guard, state or allocation work",
        ),
        (
            "shadowed guard",
            ALLOCATION.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
            "all.guard_bindings: observed 2, expected 1",
        ),
        (
            "other reset receiver",
            ALLOCATION.replacen(reset, "    other.reset_state();", 1),
            "unreviewed guard, state or allocation work",
        ),
        (
            "removed reset",
            ALLOCATION.replacen(reset, "", 1),
            "all.state_paths: observed 2, expected 3",
        ),
        (
            "duplicated reset",
            ALLOCATION.replacen(reset, &format!("{reset}\n{reset}"), 1),
            "all.state_paths: observed 4, expected 3",
        ),
        (
            "deferred reset",
            ALLOCATION.replacen(reset, "    let deferred = || state.reset_state();", 1),
            "unreviewed guard, state or allocation work",
        ),
        (
            "conditional reset",
            ALLOCATION.replacen(reset, &format!("    #[cfg(any())]\n{reset}"), 1),
            "unreviewed guard, state or allocation work",
        ),
        (
            "macro guard use",
            ALLOCATION.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
            "opaque or shadowed macro",
        ),
        (
            "other step receiver",
            ALLOCATION.replacen("state.forward_step(0, 1)", "other.forward_step(0, 1)", 1),
            "unreviewed guard, state or allocation work",
        ),
        (
            "late guard",
            ALLOCATION
                .replacen(guard, "", 1)
                .replacen(reset, &format!("{reset}\n{guard}"), 1),
            "guard must precede state",
        ),
        (
            "removed entry",
            ALLOCATION.replacen("    run();", "    return;", 1),
            "expected direct Metal entry",
        ),
    ];

    for (name, mutated, expected) in mutations {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        assert_eq!(check_allocation(&mutated), Err(expected.into()), "{name}");
    }
}

#[test]
fn shared_guard_mutations_are_rejected_in_all_profiler_variants() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    for (source, counters) in [(DECODE, false), (GDN, true)] {
        lock_contract::assert_guard_mutations(
            source,
            guard,
            guard,
            "_gpu_lock",
            [
                "unreviewed guard, state or work use",
                "all.guard_bindings: observed 2, expected 1",
                "opaque or shadowed macro",
            ],
            "state.reset_state()",
            "unreviewed guard, state or work use",
            |source| check(source, counters),
        );
    }
    lock_contract::assert_guard_mutations(
        ALLOCATION,
        guard,
        guard,
        "_gpu_lock",
        [
            "unreviewed guard, state or allocation work",
            "all.guard_bindings: observed 2, expected 1",
            "opaque or shadowed macro",
        ],
        "state.reset_state()",
        "unreviewed guard, state or allocation work",
        check_allocation,
    );
}
