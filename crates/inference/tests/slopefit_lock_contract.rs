//! Source contract for slope-fit Metal generation and reset work.

#[path = "support/lock_contract.rs"]
mod lock_contract;

use lock_contract::path_is;
use syn::visit::Visit;
use syn::{Expr, Item, Pat, Stmt};

const SOURCE: &str = include_str!("../src/bin/bench_decode_slopefit.rs");

fn direct_guard(stmt: &Stmt) -> bool {
    lock_contract::guard(stmt, "_gpu_lock", false)
}

fn expected_metal_cfg(attrs: &[syn::Attribute]) -> bool {
    lock_contract::metal_cfg(attrs, false)
}

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    lock_contract::sensitive_tokens(
        tokens,
        &[
            "metal",
            "_gpu_lock",
            "gpu_test_lock",
            "generate",
            "reset_state",
        ],
        &["metal", "_gpu_lock"],
    )
}

#[derive(Default)]
struct SlopefitPolicy {
    generate: usize,
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

type Work = lock_contract::Work<SlopefitPolicy>;

impl lock_contract::WorkPolicy for SlopefitPolicy {
    fn deferred(&mut self) -> &mut usize {
        &mut self.deferred
    }
    fn expr_assign(&mut self, assign: &syn::ExprAssign) -> bool {
        if path_is(&assign.left, &["metal"]) {
            self.assignments += 1;
            let Expr::Try(try_expr) = assign.right.as_ref() else {
                self.bad = true;
                return false;
            };
            let Expr::MethodCall(map) = try_expr.expr.as_ref() else {
                self.bad = true;
                return false;
            };
            let Expr::Call(call) = map.receiver.as_ref() else {
                self.bad = true;
                return false;
            };
            self.bad |= self.deferred > 0
                || !assign.attrs.is_empty()
                || !try_expr.attrs.is_empty()
                || !map.attrs.is_empty()
                || map.method != "map_err"
                || map.args.len() != 1
                || !call.attrs.is_empty();
            if path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]) && call.args.len() == 4 {
                self.q4_assignments += 1;
            } else if path_is(&call.func, &["MetalQwen35State", "new"]) && call.args.len() == 3 {
                self.f16_assignments += 1;
            } else {
                self.bad = true;
            }
        }
        true
    }
    fn attribute(&mut self, _attr: &syn::Attribute) {
        self.attributes += 1;
    }
    fn pat_ident(&mut self, pat: &syn::PatIdent) {
        self.metal_bindings += usize::from(pat.ident == "metal");
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
    }
    fn expr_path(&mut self, path: &syn::ExprPath) {
        if path.path.is_ident("metal") {
            self.receiver_paths += 1;
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
        if call.method == "generate" || call.method == "reset_state" {
            self.bad |=
                self.deferred > 0 || !call.attrs.is_empty() || !path_is(&call.receiver, &["metal"]);
            self.generate += usize::from(call.method == "generate");
            self.reset += usize::from(call.method == "reset_state");
        }
    }
    fn macro_action(&mut self, mac: &syn::Macro) -> lock_contract::MacroAction {
        self.bad |= sensitive(mac.tokens.clone());
        lock_contract::MacroAction::Walk
    }
}

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
    let all = lock_contract::visit_file_bodies(&file, Work::default());
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
        return Err("expected Metal main entry block".into());
    };
    let mut guarded_entry = Work::default();
    guarded_entry.visit_block(&entry_block.block);
    if main.attrs.iter().any(|attr| !attr.path().is_ident("doc"))
        || main.sig.asyncness.is_some()
        || guarded_entry.run_calls != 1
    {
        return Err("Metal main entry changed".into());
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
        return Err("expected typed Metal binding".into());
    };
    if guard_index >= metal_index {
        return Err("guard must precede state binding".into());
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
        return Err("expected Q4/f16 branch".into());
    };
    let Some((_, otherwise)) = &branch.else_branch else {
        return Err("missing f16 branch".into());
    };
    let Expr::Block(otherwise) = otherwise.as_ref() else {
        return Err("unexpected f16 branch".into());
    };
    let mut q4 = Work::default();
    q4.visit_block(&branch.then_branch);
    let mut f16 = Work::default();
    f16.visit_block(&otherwise.block);
    if q4.bad || f16.bad {
        return Err("unreviewed Q4/f16 construction".into());
    }
    lock_contract::check_counts(&[
        ("q4.assignments", q4.assignments, 1),
        ("q4.q4_assignments", q4.q4_assignments, 1),
        ("q4.f16_assignments", q4.f16_assignments, 0),
        ("f16.assignments", f16.assignments, 1),
        ("f16.q4_assignments", f16.q4_assignments, 0),
        ("f16.f16_assignments", f16.f16_assignments, 1),
    ])?;
    if all.bad || inside.bad {
        return Err("unreviewed guard, state, work, or entrypoint use".into());
    }
    lock_contract::check_counts(&[
        ("inside.attributes", inside.attributes, 0),
        ("all.guard_calls", all.guard_calls, 1),
        ("all.guard_bindings", all.guard_bindings, 1),
        ("inside.guard_bindings", inside.guard_bindings, 1),
        ("all.metal_bindings", all.metal_bindings, 1),
        ("inside.metal_bindings", inside.metal_bindings, 1),
        ("all.receiver_paths", all.receiver_paths, 6),
        ("inside.receiver_paths", inside.receiver_paths, 6),
        ("all.generate", all.generate, 2),
        ("inside.generate", inside.generate, 2),
        ("all.reset", all.reset, 2),
        ("inside.reset", inside.reset, 2),
        ("all.assignments", all.assignments, 2),
        ("inside.assignments", inside.assignments, 2),
        ("all.q4_assignments", all.q4_assignments, 1),
        ("inside.q4_assignments", inside.q4_assignments, 1),
        ("all.f16_assignments", all.f16_assignments, 1),
        ("inside.f16_assignments", inside.f16_assignments, 1),
        ("before.assignments", before.assignments, 0),
        ("before.generate", before.generate, 0),
        ("before.reset", before.reset, 0),
        ("before.metal_bindings", before.metal_bindings, 0),
        ("all.run_calls", all.run_calls, 1),
        ("entry.run_calls", entry.run_calls, 1),
    ])?;
    Ok(())
}

#[test]
fn slopefit_generate_and_resets_keep_the_measurement_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn slopefit_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "            metal.reset_state();";
    let generate =
        "            let _ = metal.generate(&prompt, &tokenizer, &make_cfg(warmup_tokens))?;";
    for (name, anchor) in [
        ("guard", guard),
        ("reset", reset),
        ("generate", generate),
        ("main", "fn main() {"),
        (
            "metal cfg",
            "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
        ),
        ("Q4 constructor", "MetalQwen35State::from_q4_dir("),
    ] {
        assert!(SOURCE.contains(anchor), "missing {name} mutation anchor");
    }

    let mutations = [
        ("hidden macro GPU work", lock_contract::hidden_gpu_work_mutant(SOURCE, "fn run(", guard), "opaque or shadowed macro"),
        ("removed guard", SOURCE.replacen(guard, "", 1), "expected one direct guard"),
        ("conditional main", SOURCE.replacen("fn main() {", "#[cfg(any())]\nfn main() {", 1), "Metal main entry changed"),
        ("changed metal cfg", lock_contract::mutate_run_cfg(SOURCE), "run cfg changed"),
        ("dropped guard", SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1), "unreviewed guard, state, work, or entrypoint use"),
        ("shadowed guard", SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1), "all.guard_bindings: observed 2, expected 1"),
        ("other reset receiver", SOURCE.replacen(reset, "            other.reset_state();", 1), "unreviewed guard, state, work, or entrypoint use"),
        ("removed reset", SOURCE.replacen(reset, "", 1), "all.receiver_paths: observed 5, expected 6"),
        ("duplicated reset", SOURCE.replacen(reset, &format!("{reset}\n{reset}"), 1), "all.receiver_paths: observed 7, expected 6"),
        ("deferred reset", SOURCE.replacen(
            reset,
            "            let deferred = || metal.reset_state();",
            1,
        ), "unreviewed guard, state, work, or entrypoint use"),
        ("conditional reset", SOURCE.replacen(reset, &format!("            #[cfg(any())]\n{reset}"), 1), "unreviewed guard, state, work, or entrypoint use"),
        ("other generate receiver", SOURCE.replacen(
            generate,
            "            let _ = other.generate(&prompt, &tokenizer, &make_cfg(warmup_tokens))?;",
            1,
        ), "unreviewed guard, state, work, or entrypoint use"),
        ("macro guard use", SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1), "opaque or shadowed macro"),
        ("captured state", SOURCE.replacen(
            reset,
            &format!("{reset}\n            println!(\"{{metal:?}}\");"),
            1,
        ), "unreviewed guard, state, work, or entrypoint use"),
        ("other Q4 constructor", SOURCE.replacen(
            "MetalQwen35State::from_q4_dir(",
            "OtherState::from_q4_dir(",
            1,
        ), "unreviewed Q4/f16 construction"),
        ("nested guard", SOURCE
            .replacen(guard, "", 1)
            .replacen(reset, &format!("{reset}\n{guard}"), 1), "expected one direct guard"),
    ];

    for (name, mutated, expected) in mutations {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        let error = check(&mutated).expect_err(name);
        assert_eq!(error, expected, "{name}");
    }
}

#[test]
fn top_level_guard_after_metal_binding_is_rejected() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    assert_eq!(SOURCE.matches(guard).count(), 1, "original guard anchor");
    let without_guard = SOURCE.replacen(guard, "", 1);
    let mutations = [(
        "guard after typed Metal binding",
        "    let mut metal: MetalQwen35State;",
        "guard must precede state binding",
    )];
    for (name, binding, expected) in mutations {
        assert_eq!(
            without_guard.matches(binding).count(),
            1,
            "{name}: binding anchor"
        );
        let mutant = without_guard.replacen(binding, &format!("{binding}\n{guard}"), 1);
        assert_eq!(mutant.matches(guard).count(), 1, "{name}: moved guard");
        let file = syn::parse_file(&mutant).expect("late-guard mutant must parse");
        let runs: Vec<_> = file
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(function) if function.sig.ident == "run" => Some(function),
                _ => None,
            })
            .collect();
        let [run] = runs.as_slice() else {
            panic!("{name}: expected one run");
        };
        let guard_index = run
            .block
            .stmts
            .iter()
            .position(direct_guard)
            .expect("direct guard");
        let state_index = run
            .block
            .stmts
            .iter()
            .position(|stmt| {
                let Stmt::Local(local) = stmt else {
                    return false;
                };
                let Pat::Type(typed) = &local.pat else {
                    return false;
                };
                matches!(typed.pat.as_ref(), Pat::Ident(binding) if binding.ident == "metal")
            })
            .expect("direct state binding");
        assert!(
            guard_index > state_index,
            "{name}: guard must follow binding"
        );
        assert_eq!(check(&mutant), Err(expected.into()), "{name}");
    }
}

#[test]
fn slopefit_real_entry_and_typed_assignment_are_recognized() {
    let file = syn::parse_file(SOURCE).unwrap();
    let main = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(f) if f.sig.ident == "main" => Some(f),
            _ => None,
        })
        .unwrap();
    assert!(main.attrs.iter().all(|attr| attr.path().is_ident("doc")));
    assert!(main.block.stmts.iter().any(|stmt| matches!(stmt,
        Stmt::Expr(Expr::Block(block), _) if expected_metal_cfg(&block.attrs))));
    let run = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(f) if f.sig.ident == "run" => Some(f),
            _ => None,
        })
        .unwrap();
    assert!(expected_metal_cfg(&run.attrs));
    assert!(run.block.stmts.iter().any(|stmt| matches!(stmt,
        Stmt::Local(local) if matches!(&local.pat, Pat::Type(typed) if matches!(typed.pat.as_ref(), Pat::Ident(id) if id.ident == "metal")) && local.init.is_none())));
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
            "unreviewed guard, state, work, or entrypoint use",
            "all.guard_bindings: observed 2, expected 1",
            "opaque or shadowed macro",
        ],
        lock_contract::GuardVector {
            expression: "metal.generate()",
            expected: "unreviewed guard, state, work, or entrypoint use",
        },
        check,
    );
}
