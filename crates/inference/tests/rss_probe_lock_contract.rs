//! Source contracts for the RSS probe’s independently guarded Metal arms.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt, Type};

const SOURCE: &str = include_str!("../examples/rss_autorelease_probe.rs");

fn path_is(expr: &Expr, names: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == names.len()
        && path.path.segments.iter().zip(names).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
        })
}

fn metal_cfg(attrs: &[syn::Attribute]) -> bool {
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
    let Ok(args) =
        all.parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
    else {
        return false;
    };
    let pair = |meta: &Meta, key: &str, value: &str| {
        matches!(meta,
        Meta::NameValue(nv) if nv.path.is_ident(key) && matches!(&nv.value,
            Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == value)))
    };
    args.len() == 2
        && pair(&args[0], "target_os", "macos")
        && pair(&args[1], "feature", "metal-gpu")
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

fn sensitive(tokens: proc_macro2::TokenStream, names: &[&str]) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => names.iter().any(|name| id == *name),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream(), names),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string()).is_ok_and(|s| {
                s.value().split('{').skip(1).any(|part| {
                    let captured: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    names.contains(&captured.as_str())
                })
            })
        }
        _ => false,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Decode,
    Gemm,
    Embed,
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
    let Type::Path(ty) = typed.ty.as_ref() else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::If(choice) = init.expr.as_ref() else {
        return false;
    };
    local.attrs.is_empty()
        && binding.ident == "state"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && ty.qself.is_none()
        && ty.path.is_ident("MetalQwen35State")
        && init.diverge.is_none()
        && choice.attrs.is_empty()
        && choice.else_branch.is_some()
}

fn pass_binding(stmt: &Stmt) -> bool {
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
        && binding.ident == "pass"
        && binding.mutability.is_some()
        && binding.by_ref.is_none()
        && binding.subpat.is_none()
        && init.diverge.is_none()
        && expect.attrs.is_empty()
        && expect.method == "expect"
        && expect.args.len() == 1
        && call.attrs.is_empty()
        && call.args.len() == 3
        && path_is(&call.func, &["MetalForwardPass", "new"])
}

fn inert_allow(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("allow")
        && attr.parse_args::<syn::Path>().is_ok_and(|path| {
            path.segments.len() == 2
                && path.segments[0].ident == "clippy"
                && path.segments[1].ident == "type_complexity"
        })
}

struct Work {
    arm: Arm,
    guards: usize,
    guard_calls: usize,
    bindings: usize,
    paths: usize,
    new: usize,
    q4: usize,
    reset: usize,
    prefill: usize,
    direct: usize,
    pooled: usize,
    pools: usize,
    attrs: usize,
    depth: usize,
    pool_scope: usize,
    deferred: usize,
    bad: bool,
}
impl Work {
    fn new(arm: Arm) -> Self {
        Self {
            arm,
            guards: 0,
            guard_calls: 0,
            bindings: 0,
            paths: 0,
            new: 0,
            q4: 0,
            reset: 0,
            prefill: 0,
            direct: 0,
            pooled: 0,
            pools: 0,
            attrs: 0,
            depth: 0,
            pool_scope: 0,
            deferred: 0,
            bad: false,
        }
    }
    fn target(&self) -> Option<&'static str> {
        match self.arm {
            Arm::Decode => Some("state"),
            Arm::Embed => Some("pass"),
            Arm::Gemm => None,
        }
    }
    fn work(&mut self) {
        if self.deferred > 0 {
            self.bad = true;
        }
        if self.depth == 0 && self.pool_scope == 0 {
            self.direct += 1;
        } else if self.depth == 1 && self.pool_scope == 1 {
            self.pooled += 1;
        } else {
            self.bad = true;
        }
    }
}
impl<'ast> Visit<'ast> for Work {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.attrs += 1;
        self.bad |= !inert_allow(attr);
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.guards += usize::from(pat.ident == "_gpu_lock");
        self.bindings += usize::from(self.target().is_some_and(|name| pat.ident == name));
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.paths += usize::from(self.target().is_some_and(|name| path.path.is_ident(name)));
        self.bad |= path.path.is_ident("_gpu_lock");
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        if self.arm == Arm::Decode {
            self.new += usize::from(path_is(&call.func, &["MetalQwen35State", "new"]));
            self.q4 += usize::from(path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]));
        } else if self.arm == Arm::Embed {
            self.new += usize::from(path_is(&call.func, &["MetalForwardPass", "new"]));
        }
        if path_is(&call.func, &["metal_matmul"]) {
            self.bad |= self.arm != Arm::Gemm || !call.attrs.is_empty() || call.args.len() != 6;
            self.work();
        }
        if path_is(&call.func, &["objc", "rc", "autoreleasepool"]) {
            self.pools += 1;
            let valid = matches!(call.args.first(), Some(Expr::Closure(closure))
                if closure.attrs.is_empty() && closure.inputs.is_empty() && closure.asyncness.is_none()
                    && closure.capture.is_none());
            self.bad |= !call.attrs.is_empty()
                || call.args.len() != 1
                || !valid
                || self.depth != 0
                || self.pool_scope != 0
                || self.deferred != 0;
            self.pool_scope += 1;
            visit::visit_expr_call(self, call);
            self.pool_scope -= 1;
            return;
        }
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        match call.method.to_string().as_str() {
            "reset_state" | "forward_prefill" if self.arm == Arm::Decode => {
                self.bad |= !path_is(&call.receiver, &["state"])
                    || !call.attrs.is_empty()
                    || self.depth != 0
                    || self.deferred != 0;
                if call.method == "reset_state" {
                    self.reset += 1;
                } else {
                    self.prefill += 1;
                }
            }
            "forward_step" => {
                self.bad |= self.arm != Arm::Decode
                    || !path_is(&call.receiver, &["state"])
                    || !call.attrs.is_empty();
                self.work();
            }
            "forward" => {
                self.bad |= self.arm != Arm::Embed
                    || !path_is(&call.receiver, &["pass"])
                    || !call.attrs.is_empty();
                self.work();
            }
            _ if self
                .target()
                .is_some_and(|name| path_is(&call.receiver, &[name])) =>
            {
                self.bad = true
            }
            _ => {}
        }
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        let allowed = self.pool_scope == 1 && self.depth == 0;
        self.depth += 1;
        if !allowed {
            self.deferred += 1;
        }
        visit::visit_expr_closure(self, closure);
        if !allowed {
            self.deferred -= 1;
        }
        self.depth -= 1;
    }
    fn visit_expr_async(&mut self, expr: &'ast syn::ExprAsync) {
        self.deferred += 1;
        visit::visit_expr_async(self, expr);
        self.deferred -= 1;
    }
    fn visit_item(&mut self, item: &'ast syn::Item) {
        self.deferred += 1;
        visit::visit_item(self, item);
        self.deferred -= 1;
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        let mut names = vec![
            "_gpu_lock",
            "gpu_test_lock",
            "metal_matmul",
            "forward_step",
            "forward_prefill",
            "reset_state",
            "forward",
            "autoreleasepool",
        ];
        if let Some(name) = self.target() {
            names.push(name);
        }
        self.bad |= sensitive(mac.tokens.clone(), &names);
    }
}
#[derive(Default)]
struct Global {
    reset: usize,
    prefill: usize,
    step: usize,
    gemm: usize,
    forward: usize,
    bad: bool,
}
impl<'ast> Visit<'ast> for Global {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.reset += usize::from(call.method == "reset_state");
        self.prefill += usize::from(call.method == "forward_prefill");
        self.step += usize::from(call.method == "forward_step");
        self.forward += usize::from(call.method == "forward");
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.gemm += usize::from(path_is(&call.func, &["metal_matmul"]));
        visit::visit_expr_call(self, call);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= sensitive(
            mac.tokens.clone(),
            &[
                "metal_matmul",
                "forward_step",
                "forward_prefill",
                "reset_state",
                "forward",
            ],
        );
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    for (name, kind) in [
        ("run_decode_arm", Arm::Decode),
        ("run_gemm_arm", Arm::Gemm),
        ("run_embed_arm", Arm::Embed),
    ] {
        let functions = file
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(f) if f.sig.ident == name => Some(f),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [function] = functions.as_slice() else {
            return Err("arm changed");
        };
        let attrs = function
            .attrs
            .iter()
            .filter(|attr| !attr.path().is_ident("doc"))
            .cloned()
            .collect::<Vec<_>>();
        if !metal_cfg(&attrs) || function.sig.asyncness.is_some() {
            return Err("arm cfg changed");
        }
        let guards = function
            .block
            .stmts
            .iter()
            .enumerate()
            .filter(|(_, stmt)| guard(stmt))
            .collect::<Vec<_>>();
        let [(g, _)] = guards.as_slice() else {
            return Err("guard missing");
        };
        if kind != Arm::Gemm {
            let bindings = function
                .block
                .stmts
                .iter()
                .enumerate()
                .filter(|(_, stmt)| {
                    if kind == Arm::Decode {
                        state_binding(stmt)
                    } else {
                        pass_binding(stmt)
                    }
                })
                .collect::<Vec<_>>();
            let [(b, _)] = bindings.as_slice() else {
                return Err("state construction changed");
            };
            if g >= b {
                return Err("guard must precede construction");
            }
            if kind == Arm::Decode {
                let mut initial = Work::new(kind);
                initial.visit_stmt(bindings[0].1);
                if initial.new != 1 || initial.q4 != 1 {
                    return Err("decode branches changed");
                }
            }
        }
        let mut all = Work::new(kind);
        all.visit_block(&function.block);
        let mut before = Work::new(kind);
        for stmt in &function.block.stmts[..*g] {
            before.visit_stmt(stmt);
        }
        let expected = match kind {
            Arm::Decode => (1, 6, 1, 1, 1, 1, 0),
            Arm::Gemm => (0, 0, 0, 0, 0, 0, 0),
            Arm::Embed => (1, 4, 1, 0, 0, 0, 1),
        };
        if all.bad
            || all.guards != 1
            || all.guard_calls != 1
            || (
                all.bindings,
                all.paths,
                all.new,
                all.q4,
                all.reset,
                all.prefill,
                all.attrs,
            ) != expected
            || all.direct != 2
            || all.pooled != 2
            || all.pools != 2
            || before.bindings != 0
            || before.paths != 0
            || before.new != 0
            || before.q4 != 0
            || before.reset != 0
            || before.prefill != 0
            || before.direct != 0
            || before.pooled != 0
        {
            return Err("unreviewed guard or work use");
        }
    }
    let mut global = Global::default();
    global.visit_file(&file);
    if global.bad
        || global.reset != 1
        || global.prefill != 1
        || global.step != 4
        || global.gemm != 4
        || global.forward != 4
    {
        return Err("work outside reviewed arms");
    }
    Ok(())
}

#[test]
fn rss_probe_arms_keep_work_under_measurement_guards() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn rss_probe_contract_rejects_real_source_mutations() {
    let guard = "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";
    let reset = "    state.reset_state();";
    let gemm = "            let _ = metal_matmul(&a, &b, &mut c, m, k, n);";
    let embed = "            let _ = pass.forward(&hidden_input, seq_len);";
    let pooled = "objc::rc::autoreleasepool(|| {\n                let _ = metal_matmul";
    let gemm_cfg = "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\nfn run_gemm_arm";
    assert!(
        SOURCE.matches(guard).count() == 3
            && SOURCE.contains(reset)
            && SOURCE.contains(gemm)
            && SOURCE.contains(embed)
            && SOURCE.contains(pooled)
            && SOURCE.contains(gemm_cfg)
    );
    for arm in ["run_decode_arm", "run_gemm_arm", "run_embed_arm"] {
        let marker = format!("fn {arm}(");
        let (prefix, suffix) = SOURCE.split_once(marker.as_str()).expect("arm source");
        assert!(suffix.contains(guard));
        let mutated = format!("{prefix}{marker}{}", suffix.replacen(guard, "", 1));
        assert!(check(&mutated).is_err());
    }
    for mutated in [
        SOURCE.replacen(guard, "", 1),
        SOURCE.replacen(guard, &format!("{guard}    drop(_gpu_lock);\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    let _gpu_lock = ();\n"), 1),
        SOURCE.replacen(guard, &format!("{guard}    consume!(_gpu_lock);\n"), 1),
        SOURCE.replacen(reset, "    other.reset_state();", 1),
        SOURCE.replacen(reset, "", 1),
        SOURCE.replacen(reset, &format!("{reset}\n{reset}"), 1),
        SOURCE.replacen(
            gemm,
            "            let _ = other_matmul(&a, &b, &mut c, m, k, n);",
            1,
        ),
        SOURCE.replacen(gemm, "", 1),
        SOURCE.replacen(gemm, &format!("{gemm}\n{gemm}"), 1),
        SOURCE.replacen(
            pooled,
            "defer_pool(|| {\n                let _ = metal_matmul",
            1,
        ),
        SOURCE.replacen(
            embed,
            "            let _ = other.forward(&hidden_input, seq_len);",
            1,
        ),
        SOURCE.replacen(
            embed,
            "            let _ = || pass.forward(&hidden_input, seq_len);",
            1,
        ),
        SOURCE.replacen(embed, &format!("            #[cfg(any())]\n{embed}"), 1),
        SOURCE.replacen(
            embed,
            &format!("{embed}\n            println!(\"{{pass:?}}\");"),
            1,
        ),
        SOURCE
            .replacen(guard, "", 1)
            .replacen(reset, &format!("{reset}\n{guard}"), 1),
        SOURCE.replacen(gemm_cfg, "#[cfg(any())]\nfn run_gemm_arm", 1),
    ] {
        assert!(check(&mutated).is_err());
    }
}
