//! Source invariants for the opt-in batch-GEMM MTP total-token cap.
//! This heuristic does not prove every emission is bounded; the ignored real Metal test
//! exercises the full path with a checkpoint.

use syn::visit::{self, Visit};
use syn::{BinOp, Expr, Item, Pat, Stmt};

fn source() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/qwen35/src/forward/metal_qwen35/inner/speculative.rs");
    std::fs::read_to_string(path).expect("read workspace MTP source")
}

fn path(expr: &Expr, name: &str) -> bool {
    matches!(expr, Expr::Path(p) if p.qself.is_none() && p.path.is_ident(name))
}

fn length(expr: &Expr) -> bool {
    matches!(expr, Expr::MethodCall(call) if call.method == "len" && call.args.is_empty() && path(&call.receiver, "generated_ids"))
}

fn compare(expr: &Expr, op: fn(&BinOp) -> bool) -> bool {
    matches!(expr, Expr::Binary(binary) if op(&binary.op) && length(&binary.left) && path(&binary.right, "decode_cap"))
}

fn cap_binding(stmt: &Stmt) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Pat::Ident(binding) = &local.pat else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    matches!(init.expr.as_ref(), Expr::MethodCall(call)
        if binding.ident == "decode_cap" && local.attrs.is_empty()
        && call.method == "effective_decode_cap" && call.args.is_empty()
        && path(&call.receiver, "gen_cfg"))
}

fn zero_guard(stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::If(branch), _) = stmt else {
        return false;
    };
    let Expr::Binary(test) = branch.cond.as_ref() else {
        return false;
    };
    if !matches!(&test.op, BinOp::Eq(_))
        || !path(&test.left, "decode_cap")
        || !matches!(test.right.as_ref(), Expr::Lit(value) if matches!(&value.lit, syn::Lit::Int(number) if number.base10_parse::<usize>().ok() == Some(0)))
        || branch.else_branch.is_some()
    {
        return false;
    }
    matches!(
        branch.then_branch.stmts.as_slice(),
        [Stmt::Expr(Expr::Return(_), Some(_))]
    )
}

#[derive(Default)]
struct LegacyBudget {
    uses: usize,
}

impl<'ast> Visit<'ast> for LegacyBudget {
    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if path(&field.base, "gen_cfg")
            && matches!(&field.member, syn::Member::Named(name) if name == "max_new_tokens")
        {
            self.uses += 1;
        }
        visit::visit_expr_field(self, field);
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source parse")?;
    let methods: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Impl(implementation) => Some(implementation),
            _ => None,
        })
        .flat_map(|implementation| &implementation.items)
        .filter_map(|item| match item {
            syn::ImplItem::Fn(method)
                if method.sig.ident == "generate_greedy_mtp_batch_gemm_legacy" =>
            {
                Some(method)
            }
            _ => None,
        })
        .collect();
    let [method] = methods.as_slice() else {
        return Err("batch method");
    };
    let mut budget = LegacyBudget::default();
    budget.visit_block(&method.block);
    if budget.uses != 0 {
        return Err("legacy max_new_tokens");
    }
    let Some(cap_index) = method.block.stmts.iter().position(cap_binding) else {
        return Err("effective cap binding");
    };
    let Some(zero_index) = method.block.stmts.iter().position(zero_guard) else {
        return Err("zero guard");
    };
    let Some(pending_index) = method.block.stmts.iter().position(|stmt| {
        matches!(stmt, Stmt::Local(local) if matches!(&local.pat, Pat::Ident(binding) if binding.ident == "pending_first"))
    }) else { return Err("pending token binding") };
    let Some(loop_index) = method.block.stmts.iter().position(|stmt| {
        matches!(stmt, Stmt::Expr(Expr::While(expr), _) if compare(&expr.cond, |op| matches!(op, BinOp::Lt(_))))
    }) else {
        return Err("bounded loop");
    };
    if cap_index >= zero_index || zero_index >= pending_index || pending_index >= loop_index {
        return Err("cap/zero guard ordering");
    }
    Ok(())
}

#[test]
fn batch_mtp_binds_effective_cap_and_guards_zero_before_sampling() {
    assert_eq!(check(&source()), Ok(()));
}

const EXAMPLE: &str = r#"
impl Example {
    fn generate_greedy_mtp_batch_gemm_legacy(&mut self) {
        let decode_cap = gen_cfg.effective_decode_cap();
        if decode_cap == 0 { return; }
        let pending_first = sample();
        let mut generated_ids = Vec::with_capacity(decode_cap);
        while generated_ids.len() < decode_cap {
            let remaining = decode_cap - generated_ids.len();
            break;
        }
    }
}
"#;

#[test]
fn batch_cap_contract_accepts_behavior_preserving_rewrites() {
    assert_eq!(check(EXAMPLE), Ok(()));
    let rewritten = EXAMPLE
        .replace("Vec::with_capacity(decode_cap)", "Vec::new()")
        .replace(
            "decode_cap - generated_ids.len()",
            "decode_cap.saturating_sub(generated_ids.len())",
        );
    assert_ne!(rewritten, EXAMPLE);
    assert_eq!(check(&rewritten), Ok(()));
}

#[test]
fn batch_cap_contract_rejects_original_cap_and_zero_guard_mutants() {
    let actual = source();
    let binding = "let decode_cap = gen_cfg.effective_decode_cap();";
    assert!(actual.contains(binding), "missing effective cap anchor");
    let original_mutant = actual.replacen(binding, "let decode_cap = gen_cfg.max_new_tokens;", 1);
    assert_eq!(check(&original_mutant), Err("legacy max_new_tokens"));

    let old_cap = EXAMPLE.replace("gen_cfg.effective_decode_cap()", "gen_cfg.max_new_tokens");
    assert_ne!(old_cap, EXAMPLE);
    assert_eq!(check(&old_cap), Err("legacy max_new_tokens"));

    let old_bound = EXAMPLE.replace(
        "generated_ids.len() < decode_cap",
        "generated_ids.len() < gen_cfg.max_new_tokens",
    );
    assert_ne!(old_bound, EXAMPLE);
    assert_eq!(check(&old_bound), Err("legacy max_new_tokens"));

    let unbounded_loop = EXAMPLE.replace("generated_ids.len() < decode_cap", "true");
    assert_ne!(unbounded_loop, EXAMPLE);
    assert_eq!(check(&unbounded_loop), Err("bounded loop"));

    let late_zero = EXAMPLE.replace(
        "if decode_cap == 0 { return; }\n        let pending_first = sample();",
        "let pending_first = sample();\n        if decode_cap == 0 { return; }",
    );
    assert_ne!(late_zero, EXAMPLE);
    assert_eq!(check(&late_zero), Err("cap/zero guard ordering"));

    let wrong_zero = EXAMPLE.replace("decode_cap == 0", "decode_cap == 1");
    assert_ne!(wrong_zero, EXAMPLE);
    assert_eq!(check(&wrong_zero), Err("zero guard"));
}
