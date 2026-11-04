//! Source contract for Q4/Q8 Metal prefill and reset work.

#[path = "support/lock_contract.rs"]
mod lock_contract;
#[path = "support/prefill_pruning.rs"]
mod prefill_pruning;

use lock_contract::path_is;
use prefill_pruning::StatePolicy;
use syn::visit::Visit;
use syn::{Expr, Item, Pat, Stmt};

const Q4: &str = include_str!("../examples/bench_q4_prefill.rs");
const Q8: &str = include_str!("../examples/bench_q8_prefill.rs");

fn direct_guard(stmt: &Stmt) -> bool {
    lock_contract::guard(stmt, "_gpu_lock", false)
}

fn direct_state(stmt: &Stmt, constructor: &str, args: usize) -> bool {
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
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && expect.attrs.is_empty()
        && expect.method == "expect"
        && expect.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == args
        && path_is(&call.func, &["MetalQwen35State", constructor])
}

fn expected_metal_cfg(attrs: &[syn::Attribute]) -> bool {
    lock_contract::metal_cfg(attrs, false)
}

type Work = lock_contract::Work<StatePolicy>;

fn check(source: &str, constructor: &str, args: usize) -> Result<(), String> {
    let file = lock_contract::check(source)?;
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
        return Err("expected one run and main".into());
    };
    if !expected_metal_cfg(&run.attrs) || run.sig.asyncness.is_some() {
        return Err("run cfg changed".into());
    }
    let guards: Vec<_> = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_guard(stmt))
        .collect();
    let [(guard_index, _)] = guards.as_slice() else {
        return Err("expected one direct guard".into());
    };
    let all = lock_contract::visit_file_bodies(&file, Work::new(StatePolicy::new(false)));
    let mut inside = Work::new(StatePolicy::new(false));
    inside.visit_block(&run.block);
    let mut before = Work::new(StatePolicy::new(false));
    for stmt in &run.block.stmts[..*guard_index] {
        before.visit_stmt(stmt);
    }
    let mut entry = Work::new(StatePolicy::new(false));
    entry.visit_block(&main.block);
    let entry_calls = main
        .block
        .stmts
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Expr(Expr::Call(call), _)
                if expected_metal_cfg(&call.attrs)
                    && path_is(&call.func, &["run"])
                    && call.args.is_empty() =>
            {
                Some(call)
            }
            _ => None,
        })
        .count();
    if main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
        || entry_calls != 1
    {
        return Err("Metal main entry changed".into());
    }
    let states = run
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| direct_state(stmt, constructor, args))
        .collect::<Vec<_>>();
    let [(state_index, _)] = states.as_slice() else {
        return Err("expected reviewed state constructor".into());
    };
    if guard_index >= state_index {
        return Err("guard must precede state".into());
    }
    if all.bad || inside.bad {
        return Err("unreviewed guard, state, work, or entrypoint use".into());
    }
    lock_contract::check_counts(&[
        ("inside.attributes", inside.attributes, 0),
        ("all.guard_calls", all.guard_calls, 1),
        ("all.guard_bindings", all.guard_bindings, 1),
        ("inside.guard_bindings", inside.guard_bindings, 1),
        ("all.state_bindings", all.state_bindings, 1),
        ("inside.state_bindings", inside.state_bindings, 1),
        ("all.receiver_paths", all.receiver_paths, 4),
        ("inside.receiver_paths", inside.receiver_paths, 4),
        ("all.prefill", all.prefill, 2),
        ("inside.prefill", inside.prefill, 2),
        ("all.reset", all.reset, 2),
        ("inside.reset", inside.reset, 2),
        ("before.prefill", before.prefill, 0),
        ("before.reset", before.reset, 0),
        ("before.state_bindings", before.state_bindings, 0),
        ("all.run_calls", all.run_calls, 1),
        ("entry.run_calls", entry.run_calls, 1),
    ])?;
    Ok(())
}

#[test]
fn q4_and_q8_prefill_resets_keep_the_measurement_guard() {
    assert_eq!(check(Q4, "from_q4_dir", 4), Ok(()));
    assert_eq!(check(Q8, "new", 3), Ok(()));
}

#[test]
fn prefill_contract_rejects_real_source_mutations() {
    for (source, constructor, args) in [(Q4, "from_q4_dir", 4), (Q8, "new", 3)] {
        let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
        let reset = "        state.reset_state();";
        let prefill = "        let _ = state.forward_prefill(&prompt_ids);";
        for (name, anchor) in [
            ("guard", guard),
            ("reset", reset),
            ("prefill", prefill),
            ("main", "fn main() {"),
            (
                "metal cfg",
                "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
            ),
        ] {
            assert!(source.contains(anchor), "missing {name} mutation anchor");
        }

        let mutations = [
            (
                "removed guard",
                source.replacen(guard, "", 1),
                "expected one direct guard",
            ),
            (
                "conditional main",
                source.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
                "Metal main entry changed",
            ),
            (
                "changed cfg",
                lock_contract::mutate_run_cfg(source),
                "run cfg changed",
            ),
            (
                "dropped guard",
                source.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
                "unreviewed guard, state, work, or entrypoint use",
            ),
            (
                "shadowed guard",
                source.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
                "all.guard_bindings: observed 2, expected 1",
            ),
            (
                "other reset receiver",
                source.replacen(reset, "        other.reset_state();", 1),
                "unreviewed guard, state, work, or entrypoint use",
            ),
            (
                "removed reset",
                source.replacen(reset, "", 1),
                "all.receiver_paths: observed 3, expected 4",
            ),
            (
                "duplicated reset",
                source.replacen(reset, &format!("{reset}\n{reset}"), 1),
                "all.receiver_paths: observed 5, expected 4",
            ),
            (
                "other prefill receiver",
                source.replacen(
                    prefill,
                    "        let _ = other.forward_prefill(&prompt_ids);",
                    1,
                ),
                "unreviewed guard, state, work, or entrypoint use",
            ),
            (
                "deferred reset",
                source.replacen(reset, "        let deferred = || state.reset_state();", 1),
                "unreviewed guard, state, work, or entrypoint use",
            ),
            (
                "macro guard use",
                source.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
                "opaque or shadowed macro",
            ),
            (
                "captured state",
                source.replacen(
                    prefill,
                    &format!("{prefill}\n        println!(\"{{state:?}}\");"),
                    1,
                ),
                "unreviewed guard, state, work, or entrypoint use",
            ),
            (
                "nested guard",
                source
                    .replacen(guard, "", 1)
                    .replacen(prefill, &format!("{prefill}\n{guard}"), 1),
                "expected one direct guard",
            ),
        ];

        for (name, mutated, expected) in mutations {
            assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
            let error = check(&mutated, constructor, args).expect_err(name);
            assert_eq!(error, expected, "{name}");
        }
    }
}

#[test]
fn doc_comment_main_with_metal_cfg_call_is_recognized() {
    let file = syn::parse_file(
        r#"
        /// Target documentation is inert.
        fn main() {
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            run();
        }
    "#,
    )
    .unwrap();
    let [Item::Fn(main)] = file.items.as_slice() else {
        panic!("expected main")
    };
    assert!(main.attrs.iter().all(|attr| attr.path().is_ident("doc")));
    let [Stmt::Expr(Expr::Call(call), Some(_))] = main.block.stmts.as_slice() else {
        panic!("expected call statement")
    };
    assert!(expected_metal_cfg(&call.attrs));
    assert!(path_is(&call.func, &["run"]));
}
