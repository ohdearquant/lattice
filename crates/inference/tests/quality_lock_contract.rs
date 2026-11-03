//! Source contract for Metal work in the layer-pruning quality benchmark.

#[path = "support/lock_contract.rs"]
mod lock_contract;

use lock_contract::path_is;
use syn::visit::Visit;
use syn::{Expr, Item, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_quality.rs");
const STATES: [&str; 3] = ["state", "state8", "state12"];

fn metal_cfg(attrs: &[syn::Attribute]) -> bool {
    lock_contract::metal_cfg(attrs, false)
}

fn direct_guard(stmt: &Stmt) -> bool {
    lock_contract::guard(stmt, "_gpu_lock", false)
}

fn state_index(expr: &Expr) -> Option<usize> {
    STATES.iter().position(|name| path_is(expr, &[*name]))
}

fn state_binding(stmt: &Stmt) -> Option<usize> {
    let Stmt::Local(local) = stmt else {
        return None;
    };
    let Pat::Ident(binding) = &local.pat else {
        return None;
    };
    let index = STATES.iter().position(|name| binding.ident == *name)?;
    let init = local.init.as_ref()?;
    let Expr::MethodCall(expect) = init.expr.as_ref() else {
        return None;
    };
    let Expr::Call(call) = expect.receiver.as_ref() else {
        return None;
    };
    (local.attrs.is_empty()
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && expect.attrs.is_empty()
        && expect.method == "expect"
        && expect.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == 4
        && path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]))
    .then_some(index)
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    lock_contract::sensitive_tokens(
        tokens,
        &[
            "state",
            "state8",
            "state12",
            "_gpu_lock",
            "gpu_test_lock",
            "reset_state",
            "forward_prefill",
            "score_layer_importance",
        ],
        &["state", "state8", "state12", "_gpu_lock"],
    )
}

#[derive(Default)]
struct StateCounts {
    state: usize,
    state8: usize,
    state12: usize,
}

impl StateCounts {
    fn increment(&mut self, index: usize) {
        match index {
            0 => self.state += 1,
            1 => self.state8 += 1,
            2 => self.state12 += 1,
            _ => unreachable!("state index is derived from STATES"),
        }
    }
}

#[derive(Default)]
struct QualityPolicy {
    bindings: StateCounts,
    paths: StateCounts,
    reset: StateCounts,
    prefill: StateCounts,
    score: StateCounts,
    drops: StateCounts,
    guard_bindings: usize,
    guard_calls: usize,
    run_calls: usize,
    attributes: usize,
    deferred: usize,
    bad: bool,
}
type Work = lock_contract::Work<QualityPolicy>;

impl lock_contract::WorkPolicy for QualityPolicy {
    fn deferred(&mut self) -> &mut usize {
        &mut self.deferred
    }
    fn attribute(&mut self, _attr: &syn::Attribute) {
        self.attributes += 1;
    }
    fn pat_ident(&mut self, pat: &syn::PatIdent) {
        if let Some(index) = STATES.iter().position(|name| pat.ident == *name) {
            self.bindings.increment(index);
        }
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
    }
    fn expr_path(&mut self, path: &syn::ExprPath) {
        if let Some(index) = STATES.iter().position(|name| path.path.is_ident(*name)) {
            self.paths.increment(index);
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
        if path_is(&call.func, &["drop"])
            && call.args.len() == 1
            && let Some(index) = call.args.first().and_then(state_index)
        {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            self.drops.increment(index);
        }
    }
    fn expr_method_call(&mut self, call: &syn::ExprMethodCall) {
        let work = call.method == "reset_state"
            || call.method == "forward_prefill"
            || call.method == "score_layer_importance";
        if let Some(index) = state_index(&call.receiver) {
            self.bad |= self.deferred > 0 || !call.attrs.is_empty();
            match call.method.to_string().as_str() {
                "reset_state" => self.reset.increment(index),
                "forward_prefill" => self.prefill.increment(index),
                "score_layer_importance" => self.score.increment(index),
                _ => self.bad = true,
            }
        } else if work {
            self.bad = true;
        }
    }
    fn macro_action(&mut self, mac: &syn::Macro) -> lock_contract::MacroAction {
        self.bad |= sensitive(mac.tokens.clone());
        lock_contract::MacroAction::Walk
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
    let runs = find("run");
    let mains = find("main");
    let ([run], [main]) = (runs.as_slice(), mains.as_slice()) else {
        return Err("expected one run and main".into());
    };
    if !metal_cfg(&run.attrs)
        || run.sig.asyncness.is_some()
        || main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
    {
        return Err("entry cfg changed".into());
    }
    let entry = main.block.stmts.iter().filter(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs) && path_is(&call.func, &["run"]) && call.args.is_empty())).count();
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
    let [(guard_index, _)] = guards.as_slice() else {
        return Err("guard missing".into());
    };
    let bindings = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(pos, stmt)| state_binding(stmt).map(|index| (pos, index)))
        .collect::<Vec<_>>();
    if bindings.len() != 3
        || bindings
            .iter()
            .enumerate()
            .any(|(i, (pos, index))| *pos <= *guard_index || *index != i)
    {
        return Err("unreviewed state construction".into());
    }
    let all = lock_contract::visit_file_bodies(&file, Work::default());
    let mut inside = Work::default();
    inside.visit_block(&run.block);
    let mut before = Work::default();
    for stmt in &run.block.stmts[..*guard_index] {
        before.visit_stmt(stmt);
    }
    if all.bad || inside.bad {
        return Err("unreviewed guard, state or work use".into());
    }
    lock_contract::check_counts(&[
        ("inside.attributes", inside.attributes, 0),
        ("all.guard_calls", all.guard_calls, 1),
        ("all.guard_bindings", all.guard_bindings, 1),
        ("inside.guard_bindings", inside.guard_bindings, 1),
        ("all.bindings.state", all.bindings.state, 1),
        ("all.bindings.state8", all.bindings.state8, 1),
        ("all.bindings.state12", all.bindings.state12, 1),
        ("inside.bindings.state", inside.bindings.state, 1),
        ("inside.bindings.state8", inside.bindings.state8, 1),
        ("inside.bindings.state12", inside.bindings.state12, 1),
        ("all.paths.state", all.paths.state, 4),
        ("all.paths.state8", all.paths.state8, 3),
        ("all.paths.state12", all.paths.state12, 3),
        ("inside.paths.state", inside.paths.state, 4),
        ("inside.paths.state8", inside.paths.state8, 3),
        ("inside.paths.state12", inside.paths.state12, 3),
        ("all.reset.state", all.reset.state, 1),
        ("all.reset.state8", all.reset.state8, 1),
        ("all.reset.state12", all.reset.state12, 1),
        ("inside.reset.state", inside.reset.state, 1),
        ("inside.reset.state8", inside.reset.state8, 1),
        ("inside.reset.state12", inside.reset.state12, 1),
        ("all.prefill.state", all.prefill.state, 1),
        ("all.prefill.state8", all.prefill.state8, 1),
        ("all.prefill.state12", all.prefill.state12, 1),
        ("inside.prefill.state", inside.prefill.state, 1),
        ("inside.prefill.state8", inside.prefill.state8, 1),
        ("inside.prefill.state12", inside.prefill.state12, 1),
        ("all.score.state", all.score.state, 1),
        ("all.score.state8", all.score.state8, 0),
        ("all.score.state12", all.score.state12, 0),
        ("inside.score.state", inside.score.state, 1),
        ("inside.score.state8", inside.score.state8, 0),
        ("inside.score.state12", inside.score.state12, 0),
        ("all.drops.state", all.drops.state, 1),
        ("all.drops.state8", all.drops.state8, 1),
        ("all.drops.state12", all.drops.state12, 1),
        ("inside.drops.state", inside.drops.state, 1),
        ("inside.drops.state8", inside.drops.state8, 1),
        ("inside.drops.state12", inside.drops.state12, 1),
        ("before.bindings.state", before.bindings.state, 0),
        ("before.bindings.state8", before.bindings.state8, 0),
        ("before.bindings.state12", before.bindings.state12, 0),
        ("before.paths.state", before.paths.state, 0),
        ("before.paths.state8", before.paths.state8, 0),
        ("before.paths.state12", before.paths.state12, 0),
        ("all.run_calls", all.run_calls, 1),
    ])?;
    Ok(())
}

#[test]
fn quality_benchmark_keeps_resets_prefill_and_scoring_under_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn quality_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "        state.reset_state();";
    let score = "        match state.score_layer_importance(&scoring_prompts, 12) {";
    for (name, anchor) in [
        ("guard", guard),
        ("reset", reset),
        ("score", score),
        ("main", "fn main() {"),
        (
            "metal cfg",
            "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
        ),
        ("state8 drop", "    drop(state8);"),
    ] {
        assert!(SOURCE.contains(anchor), "missing {name} mutation anchor");
    }
    let mutations = [
        lock_contract::hidden_gpu_work_mutant(SOURCE, "fn run(", guard),
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
        SOURCE.replacen(
            "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
            "#[cfg(any())]",
            1,
        ),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
        SOURCE.replacen(reset, "        other.reset_state();", 1),
        SOURCE.replacen(reset, "", 1),
        SOURCE.replacen(reset, &format!("{reset}\n{reset}"), 1),
        SOURCE.replacen(reset, "        let deferred = || state.reset_state();", 1),
        SOURCE.replacen(reset, &format!("        #[cfg(any())]\n{reset}"), 1),
        SOURCE.replacen(
            score,
            "        match other.score_layer_importance(&scoring_prompts, 12) {",
            1,
        ),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
        SOURCE.replacen(
            reset,
            &format!("{reset}\n        println!(\"{{state:?}}\");"),
            1,
        ),
        SOURCE.replacen("    drop(state8);", "    consume(state8);", 1),
        SOURCE
            .replacen(guard, "", 1)
            .replacen(reset, &format!("{reset}\n{guard}"), 1),
    ];
    let cases: [(&str, &[&str]); 16] = [
        ("hidden macro GPU work", &["opaque or shadowed macro"]),
        ("removed guard", &["guard missing"]),
        ("conditional main", &["entry cfg changed"]),
        (
            "changed metal cfg",
            &["entry cfg changed", "expected direct Metal entry call"],
        ),
        ("dropped guard", &["unreviewed guard, state or work use"]),
        (
            "shadowed guard",
            &["all.guard_bindings: observed 2, expected 1"],
        ),
        (
            "other reset receiver",
            &["unreviewed guard, state or work use"],
        ),
        (
            "removed reset",
            &["all.paths.state: observed 3, expected 4"],
        ),
        (
            "duplicated reset",
            &["all.paths.state: observed 5, expected 4"],
        ),
        ("deferred reset", &["unreviewed guard, state or work use"]),
        (
            "conditional reset",
            &["unreviewed guard, state or work use"],
        ),
        (
            "other score receiver",
            &["unreviewed guard, state or work use"],
        ),
        ("macro guard use", &["opaque or shadowed macro"]),
        ("captured state", &["unreviewed guard, state or work use"]),
        (
            "removed drop",
            &["all.drops.state8: observed 0, expected 1"],
        ),
        ("nested guard", &["guard missing"]),
    ];
    assert_eq!(cases.len(), mutations.len());
    for ((name, expected), mutated) in cases.into_iter().zip(mutations) {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        let error = check(&mutated).expect_err(name);
        assert!(expected.contains(&error.as_str()), "{name}: {error}");
    }
}

#[test]
fn top_level_guard_after_state_binding_is_rejected() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    assert_eq!(SOURCE.matches(guard).count(), 1, "original guard anchor");
    let without_guard = SOURCE.replacen(guard, "", 1);
    for (start, _) in without_guard.match_indices("let ") {
        for (offset, _) in without_guard[start..].match_indices(';').take(12) {
            let end = start + offset + 1;
            let Ok(stmt) = syn::parse_str::<Stmt>(&without_guard[start..end]) else {
                continue;
            };
            if state_binding(&stmt) == Some(0) {
                let mut mutant = without_guard.clone();
                mutant.insert_str(end, &format!("\n{guard}"));
                let Ok(file) = syn::parse_file(&mutant) else {
                    break;
                };
                let run = file.items.iter().find_map(|item| match item {
                    Item::Fn(function) if function.sig.ident == "run" => Some(function),
                    _ => None,
                });
                if let Some(run) = run {
                    let guard_index = run.block.stmts.iter().position(direct_guard);
                    let state_index = run
                        .block
                        .stmts
                        .iter()
                        .position(|stmt| state_binding(stmt) == Some(0));
                    if matches!((guard_index, state_index), (Some(g), Some(s)) if g > s) {
                        assert_eq!(check(&mutant), Err("unreviewed state construction".into()));
                        return;
                    }
                }
            }
            break;
        }
    }
    panic!("missing direct state binding for top-level guard mutation");
}

#[test]
fn quality_real_documented_main_and_cfg_call_are_recognized() {
    let file = syn::parse_file(SOURCE).unwrap();
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
        Stmt::Expr(Expr::Call(call), _) if metal_cfg(&call.attrs) && path_is(&call.func, &["run"]))));
}

#[test]
fn shared_guard_mutations_are_rejected() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    lock_contract::assert_guard_mutations(
        SOURCE,
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
        check,
    );
}
