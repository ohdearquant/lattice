use crate::lock_contract::{self, MacroAction, WorkPolicy, path_is};
use syn::{Expr, Pat};

pub(crate) struct StatePolicy {
    pruning: bool,
    pub(crate) prefill: usize,
    pub(crate) reset: usize,
    pub(crate) step: usize,
    pub(crate) drops: usize,
    pub(crate) constructors: usize,
    pub(crate) receiver_paths: usize,
    pub(crate) state_bindings: usize,
    pub(crate) guard_bindings: usize,
    pub(crate) guard_calls: usize,
    pub(crate) run_calls: usize,
    pub(crate) attributes: usize,
    deferred: usize,
    pub(crate) bad: bool,
}

impl StatePolicy {
    pub(crate) fn new(pruning: bool) -> Self {
        Self {
            pruning,
            prefill: 0,
            reset: 0,
            step: 0,
            drops: 0,
            constructors: 0,
            receiver_paths: 0,
            state_bindings: 0,
            guard_bindings: 0,
            guard_calls: 0,
            run_calls: 0,
            attributes: 0,
            deferred: 0,
            bad: false,
        }
    }
}

fn prefix_capture(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Group(group) => prefix_capture(group.stream()),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string())
                .is_ok_and(|s| s.value().contains("{state") || s.value().contains("{_gpu_lock"))
        }
        _ => false,
    })
}

fn sensitive(tokens: proc_macro2::TokenStream, pruning: bool) -> bool {
    let names: &[&str] = if pruning {
        &[
            "state",
            "_gpu_lock",
            "gpu_test_lock",
            "forward_prefill",
            "forward_step",
            "reset_state",
        ]
    } else {
        &[
            "state",
            "_gpu_lock",
            "gpu_test_lock",
            "forward_prefill",
            "reset_state",
        ]
    };
    lock_contract::sensitive_tokens(tokens.clone(), names, &["state", "_gpu_lock"])
        || prefix_capture(tokens)
}

impl WorkPolicy for StatePolicy {
    fn deferred(&mut self) -> &mut usize {
        &mut self.deferred
    }

    fn local(&mut self, local: &syn::Local) {
        if self.pruning && matches!(&local.pat, Pat::Ident(binding) if binding.ident == "state") {
            let valid = matches!(&local.pat, Pat::Ident(binding) if binding.mutability.is_some() && binding.by_ref.is_none() && binding.subpat.is_none())
                && local.attrs.is_empty()
                && matches!(&local.init, Some(init) if init.diverge.is_none() && matches!(init.expr.as_ref(), Expr::Match(expr)
                    if expr.attrs.is_empty() && matches!(expr.expr.as_ref(), Expr::Call(call)
                        if call.attrs.is_empty() && call.args.len() == 4 && path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]))));
            self.bad |= !valid || self.deferred > 0;
        }
    }

    fn attribute(&mut self, _attr: &syn::Attribute) {
        self.attributes += 1;
    }

    fn pat_ident(&mut self, pat: &syn::PatIdent) {
        self.state_bindings += usize::from(pat.ident == "state");
        self.guard_bindings += usize::from(pat.ident == "_gpu_lock");
    }

    fn expr_path(&mut self, path: &syn::ExprPath) {
        self.receiver_paths += usize::from(path.path.is_ident("state"));
        self.bad |= path.path.is_ident("_gpu_lock");
    }

    fn expr_call(&mut self, call: &syn::ExprCall) {
        self.guard_calls += usize::from(path_is(
            &call.func,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        ));
        self.run_calls += usize::from(path_is(&call.func, &["run"]));
        if self.pruning {
            self.constructors +=
                usize::from(path_is(&call.func, &["MetalQwen35State", "from_q4_dir"]));
            if path_is(&call.func, &["drop"])
                && call.args.len() == 1
                && call
                    .args
                    .first()
                    .is_some_and(|arg| path_is(arg, &["state"]))
            {
                self.bad |= self.deferred > 0 || !call.attrs.is_empty();
                self.drops += 1;
            }
        }
    }

    fn expr_method_call(&mut self, call: &syn::ExprMethodCall) {
        if call.method == "forward_prefill"
            || call.method == "reset_state"
            || (self.pruning && call.method == "forward_step")
        {
            self.bad |=
                self.deferred > 0 || !call.attrs.is_empty() || !path_is(&call.receiver, &["state"]);
            self.prefill += usize::from(call.method == "forward_prefill");
            self.reset += usize::from(call.method == "reset_state");
            self.step += usize::from(call.method == "forward_step");
        }
    }

    fn macro_action(&mut self, mac: &syn::Macro) -> MacroAction {
        self.bad |= sensitive(mac.tokens.clone(), self.pruning);
        MacroAction::Walk
    }
}

#[cfg(test)]
mod tests {
    use super::{StatePolicy, sensitive};
    use crate::lock_contract::Work;
    use syn::visit::Visit;

    #[test]
    fn pruning_keeps_step_sensitivity_and_prefill_does_not_add_it() {
        let expr: syn::Expr = syn::parse_str("state.forward_step(1, 1)").expect("step fixture");
        let mut prefill = Work::new(StatePolicy::new(false));
        prefill.visit_expr(&expr);
        let mut pruning = Work::new(StatePolicy::new(true));
        pruning.visit_expr(&expr);
        assert_eq!(prefill.step, 0);
        assert_eq!(pruning.step, 1);
    }

    #[test]
    fn legacy_prefix_capture_and_pruning_local_audit_remain() {
        let tokens = r#""{state_bytes}""#.parse().unwrap();
        assert!(sensitive(tokens, false));
        let file = syn::parse_file("fn run() { let state = 0; }").unwrap();
        let [syn::Item::Fn(run)] = file.items.as_slice() else {
            panic!("expected one function");
        };
        let mut prefill = Work::new(StatePolicy::new(false));
        prefill.visit_block(&run.block);
        let mut pruning = Work::new(StatePolicy::new(true));
        pruning.visit_block(&run.block);
        assert!(!prefill.bad);
        assert!(pruning.bad);

        let valid: syn::Block = syn::parse_str(
            "{ let mut state = match MetalQwen35State::from_q4_dir(a, b, c, d) { Ok(value) => value, Err(_) => return, }; }",
        )
        .expect("constructor fixture");
        let mut pruning = Work::new(StatePolicy::new(true));
        pruning.visit_block(&valid);
        assert!(!pruning.bad);
    }
}
