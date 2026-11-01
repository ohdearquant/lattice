//! Source contract for the opt-in batch-GEMM MTP total-token cap.

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
struct Budget {
    legacy: usize,
    cap_uses: usize,
    capacity: usize,
    remaining: usize,
    loops: usize,
    exhausted: usize,
}

impl<'ast> Visit<'ast> for Budget {
    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if path(&field.base, "gen_cfg")
            && matches!(&field.member, syn::Member::Named(name) if name == "max_new_tokens")
        {
            self.legacy += 1;
        }
        visit::visit_expr_field(self, field);
    }
    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        self.cap_uses += usize::from(expr.path.is_ident("decode_cap"));
        visit::visit_expr_path(self, expr);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let Expr::Path(func) = call.func.as_ref()
            && func
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                == ["Vec", "with_capacity"]
            && call.args.len() == 1
            && path(&call.args[0], "decode_cap")
        {
            self.capacity += 1;
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if matches!(&local.pat, Pat::Ident(binding) if binding.ident == "remaining")
            && let Some(init) = &local.init
            && let Expr::Binary(expr) = init.expr.as_ref()
            && matches!(&expr.op, BinOp::Sub(_))
            && path(&expr.left, "decode_cap")
            && length(&expr.right)
        {
            self.remaining += 1;
        }
        visit::visit_local(self, local);
    }
    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        if compare(&expr.cond, |op| matches!(op, BinOp::Lt(_))) {
            self.loops += 1;
        }
        visit::visit_expr_while(self, expr);
    }
    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        if compare(&expr.cond, |op| matches!(op, BinOp::Ge(_))) {
            self.exhausted += 1;
        }
        visit::visit_expr_if(self, expr);
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
    let mut budget = Budget::default();
    budget.visit_block(&method.block);
    if budget.legacy != 0 {
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
    if cap_index >= zero_index || zero_index >= pending_index {
        return Err("zero guard ordering");
    }
    if budget.cap_uses != 6
        || budget.capacity != 1
        || budget.remaining != 2
        || budget.loops != 1
        || budget.exhausted != 1
    {
        return Err("batch budget sites");
    }
    Ok(())
}

#[test]
fn batch_mtp_uses_effective_cap_at_each_budget_site() {
    assert_eq!(check(&source()), Ok(()));
}

#[test]
fn batch_cap_contract_rejects_old_bounds_and_zero_guard_mutations() {
    let source = source();
    for before in [
        "if decode_cap == 0",
        "Vec::with_capacity(decode_cap)",
        "while generated_ids.len() < decode_cap",
        "let remaining = decode_cap - generated_ids.len()",
        "if generated_ids.len() >= decode_cap",
    ] {
        assert!(source.contains(before), "missing anchor {before}");
        let mutant = source.replacen(
            before,
            &before.replace("decode_cap", "gen_cfg.max_new_tokens"),
            1,
        );
        assert!(syn::parse_file(&mutant).is_ok());
        assert_eq!(check(&mutant), Err("legacy max_new_tokens"), "{before}");
    }
    let remaining = "let remaining = decode_cap - generated_ids.len()";
    assert_eq!(source.matches(remaining).count(), 2);
    let mut last = source.clone();
    let offset = last.rfind(remaining).expect("second remaining site");
    last.replace_range(
        offset..offset + remaining.len(),
        &remaining.replace("decode_cap", "gen_cfg.max_new_tokens"),
    );
    assert!(syn::parse_file(&last).is_ok());
    assert_eq!(check(&last), Err("legacy max_new_tokens"));
    let zero_start = source
        .find("        if decode_cap == 0 {")
        .expect("zero guard");
    let zero_end = source[zero_start..]
        .find("        // Greedy argmax")
        .expect("post-guard anchor")
        + zero_start;
    let block = &source[zero_start..zero_end];
    let pending =
        "        let pending_first = crate::sampling::argmax_f32_first_wins(prefill_logits);";
    assert!(source.contains(pending));
    let late = source
        .replacen(block, "", 1)
        .replacen(pending, &[pending, "\n", block].concat(), 1);
    assert!(syn::parse_file(&late).is_ok());
    assert_eq!(check(&late), Err("zero guard ordering"));
    let wrong_zero = source.replacen("if decode_cap == 0", "if decode_cap == 1", 1);
    assert!(syn::parse_file(&wrong_zero).is_ok());
    assert_eq!(check(&wrong_zero), Err("zero guard"));
}
