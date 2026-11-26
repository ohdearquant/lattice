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

fn remaining_value(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(binary) => {
            matches!(&binary.op, BinOp::Sub(_))
                && path(&binary.left, "decode_cap")
                && length(&binary.right)
        }
        Expr::MethodCall(call) => {
            call.method == "saturating_sub"
                && path(&call.receiver, "decode_cap")
                && call.args.len() == 1
                && length(&call.args[0])
        }
        _ => false,
    }
}

fn named_length(expr: &Expr, name: &str) -> bool {
    matches!(expr, Expr::MethodCall(call) if call.method == "len" && call.args.is_empty() && path(&call.receiver, name))
}

fn clamped_length(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::MethodCall(call) => {
            call.method == "min"
                && named_length(&call.receiver, name)
                && call.args.len() == 1
                && path(&call.args[0], "remaining")
        }
        Expr::Call(call) => {
            matches!(call.func.as_ref(), Expr::Path(func) if func.path.segments.last().is_some_and(|segment| segment.ident == "min"))
                && call.args.len() == 2
                && ((named_length(&call.args[0], name) && path(&call.args[1], "remaining"))
                    || (path(&call.args[0], "remaining") && named_length(&call.args[1], name)))
        }
        _ => false,
    }
}

#[derive(Default)]
struct EmissionAudit<'a> {
    name: &'a str,
    remaining: usize,
    invalid_remaining: bool,
    emit_len_clamped: bool,
    extensions: usize,
    clamped_extensions: usize,
    indirect_extensions: usize,
}

impl<'ast> Visit<'ast> for EmissionAudit<'_> {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Pat::Ident(binding) = &local.pat {
            if binding.ident == "remaining" {
                self.remaining += 1;
                self.invalid_remaining |= !local
                    .init
                    .as_ref()
                    .is_some_and(|init| remaining_value(&init.expr));
            } else if binding.ident == "emit_len" {
                self.emit_len_clamped |= local
                    .init
                    .as_ref()
                    .is_some_and(|init| clamped_length(&init.expr, self.name));
            }
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "extend_from_slice" && path(&call.receiver, "generated_ids") {
            self.extensions += 1;
            if call.args.len() == 1
                && let Expr::Reference(reference) = &call.args[0]
                && let Expr::Index(index) = reference.expr.as_ref()
                && path(&index.expr, self.name)
                && let Expr::Range(range) = index.index.as_ref()
                && range.start.is_none()
                && matches!(&range.limits, syn::RangeLimits::HalfOpen(_))
                && let Some(end) = &range.end
            {
                if clamped_length(end, self.name) {
                    self.clamped_extensions += 1;
                } else if path(end, "emit_len") {
                    self.indirect_extensions += 1;
                }
            }
        }
        visit::visit_expr_method_call(self, call);
    }
}

#[derive(Default)]
struct RoundAudit {
    stop_ok: bool,
    continue_ok: bool,
    exhausted: bool,
    invalid_remaining: bool,
    invalid_capacity: bool,
}

impl<'ast> Visit<'ast> for RoundAudit {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if matches!(&local.pat, Pat::Ident(binding) if binding.ident == "remaining") {
            self.invalid_remaining |= !local
                .init
                .as_ref()
                .is_some_and(|init| remaining_value(&init.expr));
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if matches!(call.func.as_ref(), Expr::Path(func) if func.path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>() == ["Vec", "with_capacity"])
        {
            self.invalid_capacity |= call.args.len() != 1 || !path(&call.args[0], "decode_cap");
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_if(&mut self, branch: &'ast syn::ExprIf) {
        if compare(&branch.cond, |op| matches!(op, BinOp::Ge(_)))
            && branch.else_branch.is_none()
            && matches!(
                branch.then_branch.stmts.as_slice(),
                [Stmt::Expr(Expr::Break(_), _)]
            )
        {
            self.exhausted = true;
        }
        visit::visit_expr_if(self, branch);
    }

    fn visit_arm(&mut self, arm: &'ast syn::Arm) {
        let (name, stop) = match &arm.pat {
            Pat::TupleStruct(pat)
                if pat
                    .path
                    .segments
                    .last()
                    .is_some_and(|s| s.ident == "EmitAndStop") =>
            {
                ("tokens", true)
            }
            Pat::Struct(pat)
                if pat
                    .path
                    .segments
                    .last()
                    .is_some_and(|s| s.ident == "EmitAndContinue") =>
            {
                ("emit", false)
            }
            _ => return visit::visit_arm(self, arm),
        };
        let mut audit = EmissionAudit {
            name,
            ..EmissionAudit::default()
        };
        audit.visit_expr(&arm.body);
        let safe = audit.remaining > 0
            && !audit.invalid_remaining
            && audit.extensions > 0
            && audit.clamped_extensions
                + if audit.emit_len_clamped {
                    audit.indirect_extensions
                } else {
                    0
                }
                == audit.extensions;
        if stop {
            self.stop_ok |= safe;
        } else {
            self.continue_ok |= safe;
        }
        visit::visit_arm(self, arm);
    }
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
    let mut audit = RoundAudit::default();
    audit.visit_block(&method.block);
    if audit.invalid_remaining {
        return Err("remaining budget");
    }
    if audit.invalid_capacity {
        return Err("capacity budget");
    }
    let Stmt::Expr(Expr::While(loop_expr), _) = &method.block.stmts[loop_index] else {
        return Err("bounded loop");
    };
    let mut loop_audit = RoundAudit::default();
    loop_audit.visit_block(&loop_expr.body);
    if !loop_audit.exhausted {
        return Err("exhaustion guard");
    }
    if !loop_audit.stop_ok || !loop_audit.continue_ok {
        return Err("emission clamp");
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
            match outcome {
                MtpRoundOutcome::EmitAndStop(tokens) => {
                    let remaining = decode_cap - generated_ids.len();
                    let emit_len = tokens.len().min(remaining);
                    generated_ids.extend_from_slice(&tokens[..emit_len]);
                    break;
                }
                MtpRoundOutcome::EmitAndContinue { emit, .. } => {
                    let remaining = decode_cap - generated_ids.len();
                    generated_ids.extend_from_slice(&emit[..emit.len().min(remaining)]);
                    if generated_ids.len() >= decode_cap { break; }
                }
            }
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

    let actual = source();
    let rewritten = mutate(&actual, "Vec::with_capacity(decode_cap)", "Vec::new()", 0)
        .replace(
            "decode_cap - generated_ids.len()",
            "decode_cap.saturating_sub(generated_ids.len())",
        )
        .replace(
            "tokens.len().min(remaining)",
            "std::cmp::min(tokens.len(), remaining)",
        )
        .replace(
            "emit.len().min(remaining)",
            "std::cmp::min(remaining, emit.len())",
        );
    assert_ne!(rewritten, actual);
    assert_eq!(check(&rewritten), Ok(()));
}

#[test]
fn batch_cap_contract_rejects_original_cap_and_zero_guard_mutants() {
    let actual = source();
    reject_mutant(
        &actual,
        "let decode_cap = gen_cfg.effective_decode_cap();",
        "let decode_cap = gen_cfg.max_new_tokens;",
        0,
        "legacy max_new_tokens",
    );
    reject_mutant(
        &actual,
        "while generated_ids.len() < decode_cap",
        "while generated_ids.len() < gen_cfg.max_new_tokens",
        0,
        "legacy max_new_tokens",
    );
    reject_mutant(
        &actual,
        "while generated_ids.len() < decode_cap",
        "while true",
        0,
        "bounded loop",
    );
    reject_mutant(
        &actual,
        "if decode_cap == 0",
        "if decode_cap == 1",
        0,
        "zero guard",
    );
    for occurrence in 0..2 {
        reject_mutant(
            &actual,
            "let remaining = decode_cap - generated_ids.len();",
            "let remaining = 4096 - generated_ids.len();",
            occurrence,
            "remaining budget",
        );
    }
    reject_mutant(
        &actual,
        "Vec::with_capacity(decode_cap)",
        "Vec::with_capacity(4096)",
        0,
        "capacity budget",
    );
    reject_mutant(
        &actual,
        "if generated_ids.len() >= decode_cap",
        "if false",
        0,
        "exhaustion guard",
    );
    reject_mutant(
        &actual,
        "let emit_len = tokens.len().min(remaining);",
        "let emit_len = tokens.len();",
        0,
        "emission clamp",
    );
    reject_mutant(
        &actual,
        "&emit[..emit.len().min(remaining)]",
        "&emit[..emit.len()]",
        0,
        "emission clamp",
    );
}

fn reject_mutant(source: &str, before: &str, after: &str, occurrence: usize, error: &str) {
    let mutant = mutate(source, before, after, occurrence);
    assert_eq!(check(&mutant), Err(error), "mutation: {before:?}");
}

fn mutate(source: &str, before: &str, after: &str, occurrence: usize) -> String {
    let (offset, _) = source
        .match_indices(before)
        .nth(occurrence)
        .unwrap_or_else(|| panic!("missing mutation anchor {before:?} occurrence {occurrence}"));
    let mut mutant = source.to_owned();
    mutant.replace_range(offset..offset + before.len(), after);
    assert_ne!(mutant, source, "mutation did not change source");
    mutant
}
