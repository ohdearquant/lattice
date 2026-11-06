//! Source contract for the Metal layer-pruning benchmark.

#[path = "support/lock_contract.rs"]
mod lock_contract;
#[path = "support/prefill_pruning.rs"]
mod prefill_pruning;

use lock_contract::path_is;
use prefill_pruning::StatePolicy;
use syn::visit::Visit;
use syn::{Expr, Item, Stmt};

const SOURCE: &str = include_str!("../examples/bench_pruning.rs");

fn direct_guard(stmt: &Stmt) -> bool {
    lock_contract::guard(stmt, "_gpu_lock", false)
}

fn expected_metal_cfg(attrs: &[syn::Attribute]) -> bool {
    lock_contract::metal_cfg(attrs, false)
}

type Work = lock_contract::Work<StatePolicy>;

fn check(source: &str) -> Result<(), String> {
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
    let all = lock_contract::visit_file_bodies(&file, Work::new(StatePolicy::new(true)));
    let mut inside = Work::new(StatePolicy::new(true));
    inside.visit_block(&run.block);
    let mut before = Work::new(StatePolicy::new(true));
    for stmt in &run.block.stmts[..*guard_index] {
        before.visit_stmt(stmt);
    }
    let mut entry = Work::new(StatePolicy::new(true));
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
        ("all.receiver_paths", all.receiver_paths, 11),
        ("inside.receiver_paths", inside.receiver_paths, 11),
        ("all.prefill", all.prefill, 4),
        ("inside.prefill", inside.prefill, 4),
        ("all.reset", all.reset, 4),
        ("inside.reset", inside.reset, 4),
        ("all.step", all.step, 2),
        ("inside.step", inside.step, 2),
        ("all.drops", all.drops, 1),
        ("inside.drops", inside.drops, 1),
        ("all.constructors", all.constructors, 1),
        ("inside.constructors", inside.constructors, 1),
        ("before.step", before.step, 0),
        ("before.prefill", before.prefill, 0),
        ("before.reset", before.reset, 0),
        ("before.state_bindings", before.state_bindings, 0),
        ("all.run_calls", all.run_calls, 1),
        ("entry.run_calls", entry.run_calls, 1),
    ])?;
    Ok(())
}

#[test]
fn pruning_prefill_decode_and_resets_keep_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn pruning_guard_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "            state.reset_state();";
    let prefill = "            let _ = state.forward_prefill(&prompt_ids);";
    let step = "                let _ = state.forward_step(100u32 + step as u32, 32 + step);";
    for (name, anchor) in [
        ("guard", guard),
        ("reset", reset),
        ("prefill", prefill),
        ("step", step),
        ("main", "fn main() {"),
        (
            "metal cfg",
            "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
        ),
        ("state drop", "        drop(state);"),
    ] {
        assert!(SOURCE.contains(anchor), "missing {name} mutation anchor");
    }

    let mutations = [
        (
            "removed guard",
            SOURCE.replacen(guard, "", 1),
            "expected one direct guard",
        ),
        (
            "conditional main",
            SOURCE.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1),
            "Metal main entry changed",
        ),
        (
            "changed cfg",
            lock_contract::mutate_run_cfg(SOURCE),
            "run cfg changed",
        ),
        (
            "dropped guard",
            SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
            "unreviewed guard, state, work, or entrypoint use",
        ),
        (
            "shadowed guard",
            SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
            "all.guard_bindings: observed 2, expected 1",
        ),
        (
            "other reset receiver",
            SOURCE.replacen(reset, "            other.reset_state();", 1),
            "unreviewed guard, state, work, or entrypoint use",
        ),
        (
            "removed reset",
            SOURCE.replacen(reset, "", 1),
            "all.receiver_paths: observed 10, expected 11",
        ),
        (
            "conditional reset",
            SOURCE.replacen(reset, &format!("            #[cfg(any())]\n{reset}"), 1),
            "unreviewed guard, state, work, or entrypoint use",
        ),
        (
            "duplicated reset",
            SOURCE.replacen(reset, &format!("{reset}\n{reset}"), 1),
            "all.receiver_paths: observed 12, expected 11",
        ),
        (
            "other prefill receiver",
            SOURCE.replacen(
                prefill,
                "            let _ = other.forward_prefill(&prompt_ids);",
                1,
            ),
            "unreviewed guard, state, work, or entrypoint use",
        ),
        (
            "deferred step",
            SOURCE.replacen(
                step,
                "                let deferred = || state.forward_step(1, 1);",
                1,
            ),
            "unreviewed guard, state, work, or entrypoint use",
        ),
        (
            "macro guard use",
            SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
            "opaque or shadowed macro",
        ),
        (
            "captured state",
            SOURCE.replacen(
                prefill,
                &format!("{prefill}\n            println!(\"{{state:?}}\");"),
                1,
            ),
            "unreviewed guard, state, work, or entrypoint use",
        ),
        (
            "removed drop",
            SOURCE.replacen("        drop(state);", "        consume(state);", 1),
            "all.drops: observed 0, expected 1",
        ),
        (
            "nested guard",
            SOURCE
                .replacen(guard, "", 1)
                .replacen(prefill, &format!("{prefill}\n{guard}"), 1),
            "expected one direct guard",
        ),
    ];

    for (name, mutated, expected) in mutations {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        let error = check(&mutated).expect_err(name);
        assert_eq!(error, expected, "{name}");
    }
}

#[test]
fn documented_main_with_metal_cfg_call_is_recognized() {
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
