//! Selected dispatch and closure lifetime contract for the decode profiler.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/profile_metal_decode.rs");

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
        matches!(meta, Meta::NameValue(nv) if nv.path.is_ident(key) && matches!(&nv.value,
            Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Str(s) if s.value() == value)))
    };
    args.len() == 2
        && pair(&args[0], "target_os", "macos")
        && pair(&args[1], "feature", "metal-gpu")
}

fn is_guard(stmt: &Stmt) -> bool {
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

fn sensitive(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => matches!(
            id.to_string().as_str(),
            "_gpu_lock"
                | "gpu_test_lock"
                | "dispatch_ref"
                | "dispatch_flash"
                | "new_command_buffer"
                | "new_compute_command_encoder"
                | "dispatch_thread_groups"
                | "end_encoding"
                | "commit"
                | "wait_until_completed"
        ),
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        _ => false,
    })
}

#[derive(Default)]
struct Work {
    error: bool,
    closure_depth: usize,
    owner: Option<usize>,
    definitions: [usize; 2],
    calls: [usize; 2],
    // command buffer, encoder, dispatch, end encoding, commit, wait
    methods: [[usize; 6]; 2],
}

impl<'ast> Visit<'ast> for Work {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        let Pat::Ident(binding) = &local.pat else {
            visit::visit_local(self, local);
            return;
        };
        if binding.ident == "_gpu_lock" {
            self.error = true;
        }
        let owner = if binding.ident == "dispatch_ref" {
            Some(0)
        } else if binding.ident == "dispatch_flash" {
            Some(1)
        } else {
            None
        };
        if let Some(owner) = owner {
            let Some(init) = &local.init else {
                self.error = true;
                return;
            };
            let Expr::Closure(closure) = init.expr.as_ref() else {
                self.error = true;
                return;
            };
            if !local.attrs.is_empty()
                || binding.mutability.is_some()
                || binding.by_ref.is_some()
                || binding.subpat.is_some()
                || init.diverge.is_some()
                || !closure.attrs.is_empty()
                || closure.capture.is_some()
                || closure.asyncness.is_some()
                || self.closure_depth != 0
                || self.owner.is_some()
            {
                self.error = true;
            }
            self.definitions[owner] += 1;
            self.owner = Some(owner);
            self.closure_depth += 1;
            self.visit_expr(&closure.body);
            self.closure_depth -= 1;
            self.owner = None;
        } else {
            visit::visit_local(self, local);
        }
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.closure_depth += 1;
        visit::visit_expr_closure(self, closure);
        self.closure_depth -= 1;
    }

    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        if matches!(
            pat.ident.to_string().as_str(),
            "_gpu_lock" | "dispatch_ref" | "dispatch_flash"
        ) {
            self.error = true;
        }
        visit::visit_pat_ident(self, pat);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.segments.iter().any(|part| {
            matches!(
                part.ident.to_string().as_str(),
                "_gpu_lock" | "gpu_test_lock" | "dispatch_ref" | "dispatch_flash"
            )
        }) {
            self.error = true;
        }
        visit::visit_expr_path(self, path);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let owner = if path_is(&call.func, &["dispatch_ref"]) {
            Some(0)
        } else if path_is(&call.func, &["dispatch_flash"]) {
            Some(1)
        } else {
            None
        };
        if let Some(owner) = owner {
            let expected = if owner == 0 { "out_ref" } else { "out_new" };
            let valid_arg = matches!(call.args.first(), Some(Expr::Reference(reference))
                if reference.attrs.is_empty() && reference.mutability.is_none()
                    && path_is(&reference.expr, &[expected]));
            if !call.attrs.is_empty()
                || self.closure_depth != 0
                || call.args.len() != 1
                || !valid_arg
            {
                self.error = true;
            }
            self.calls[owner] += 1;
            for arg in &call.args {
                self.visit_expr(arg);
            }
        } else {
            visit::visit_expr_call(self, call);
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let (index, receiver) = match call.method.to_string().as_str() {
            "new_command_buffer" => (Some(0), "queue"),
            "new_compute_command_encoder" => (Some(1), "cmd"),
            "dispatch_thread_groups" => (Some(2), "enc"),
            "end_encoding" => (Some(3), "enc"),
            "commit" => (Some(4), "cmd"),
            "wait_until_completed" => (Some(5), "cmd"),
            _ => (None, ""),
        };
        if let Some(index) = index {
            if let Some(owner) = self.owner {
                if self.closure_depth != 1
                    || !call.attrs.is_empty()
                    || !path_is(&call.receiver, &[receiver])
                {
                    self.error = true;
                }
                self.methods[owner][index] += 1;
            } else {
                self.error = true;
            }
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        if attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr") {
            self.error = true;
        }
        visit::visit_attribute(self, attr);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if sensitive(mac.tokens.clone()) {
            self.error = true;
        }
        visit::visit_macro(self, mac);
    }
}

fn check(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "parse")?;
    let mains: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "main" && metal_cfg(&function.attrs) => {
                Some(function)
            }
            _ => None,
        })
        .collect();
    let [main] = mains.as_slice() else {
        return Err("Metal main");
    };
    let guards: Vec<_> = main
        .block
        .stmts
        .iter()
        .enumerate()
        .filter(|(_, stmt)| is_guard(stmt))
        .map(|(index, _)| index)
        .collect();
    let [guard] = guards.as_slice() else {
        return Err("guard");
    };
    if !main.block.stmts[..*guard]
        .iter()
        .all(|stmt| matches!(stmt, Stmt::Item(Item::Use(_))))
    {
        return Err("guard order");
    }
    let mut work = Work::default();
    for stmt in &main.block.stmts[*guard + 1..] {
        work.visit_stmt(stmt);
    }
    if work.error
        || work.definitions != [1, 1]
        || work.calls != [3, 3]
        || work.methods != [[1, 1, 1, 1, 1, 1], [1, 1, 3, 2, 2, 2]]
    {
        return Err("work");
    }
    Ok(())
}

#[test]
fn decode_profiler_dispatch_closures_keep_the_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn decode_profiler_contract_rejects_source_mutations() {
    let mutations = [
        (
            "let _gpu_lock = lattice_inference::measurement::gpu_test_lock();",
            "let _gpu_lock = ();",
        ),
        (
            "let _gpu_lock = lattice_inference::measurement::gpu_test_lock();",
            "let _gpu_lock = lattice_inference::measurement::gpu_test_lock(); drop(_gpu_lock);",
        ),
        (
            "let dispatch_ref = |out: &Buffer| -> f64",
            "let dispatch_ref = move |out: &Buffer| -> f64",
        ),
        (
            "let _ = dispatch_ref(&out_ref);",
            "let alias = dispatch_ref; let _ = alias(&out_ref);",
        ),
        (
            "let _ = dispatch_flash(&out_new);",
            "let later = || dispatch_flash(&out_new); let _ = later();",
        ),
        (
            "enc.dispatch_thread_groups(MTLSize::new(nqh as u64, 1, 1), MTLSize::new(256, 1, 1));",
            "enc.end_encoding();",
        ),
        ("cmd.wait_until_completed();", "cmd.commit();"),
        (
            "let _ = dispatch_ref(&out_ref);",
            "macro_rules! bypass { () => { dispatch_ref(&out_ref) } } let _ = bypass!();",
        ),
        ("for cfg in cfgs {", "#[cfg(any())] for cfg in cfgs {"),
    ];
    for (old, new) in mutations {
        assert!(SOURCE.contains(old), "missing mutation anchor: {old}");
        let mutated = SOURCE.replacen(old, new, 1);
        assert!(check(&mutated).is_err(), "mutation was accepted: {old}");
    }
}
