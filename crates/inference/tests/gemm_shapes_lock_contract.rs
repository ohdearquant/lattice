//! Selected guarded call-chain contract for the GEMM-shapes Metal example.

use syn::visit::{self, Visit};
use syn::{Expr, Item, Meta, Pat, Stmt};

const SOURCE: &str = include_str!("../examples/bench_gemm_shapes.rs");
const HELPERS: [&str; 5] = [
    "clock_warmup",
    "run_matmul_bt",
    "run_q4_tiled",
    "run_f32_sg",
    "run_attention",
];

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
        && binding.mutability.is_none()
        && binding.by_ref.is_none()
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
        proc_macro2::TokenTree::Ident(id) => {
            let id = id.to_string();
            matches!(
                id.as_str(),
                "_gpu_lock"
                    | "gpu_test_lock"
                    | "run"
                    | "timed_dispatch"
                    | "encode"
                    | "new_command_buffer"
                    | "new_compute_command_encoder"
                    | "dispatch_thread_groups"
                    | "end_encoding"
                    | "commit"
                    | "wait_until_completed"
            ) || HELPERS.contains(&id.as_str())
        }
        proc_macro2::TokenTree::Group(group) => sensitive(group.stream()),
        _ => false,
    })
}

#[derive(Default)]
struct Audit {
    current: String,
    closure_depth: usize,
    nested_fn: usize,
    error: bool,
    run_calls: usize,
    helpers: [usize; 5],
    closures: [usize; 5],
    timings: [usize; 5],
    dispatches: [usize; 5],
    // new buffer, new encoder, callback, end, commit, wait
    timed_work: [usize; 6],
}

impl Audit {
    fn helper(&self) -> Option<usize> {
        HELPERS.iter().position(|name| *name == self.current)
    }
}

impl<'ast> Visit<'ast> for Audit {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.nested_fn += 1;
        visit::visit_item_fn(self, node);
        self.nested_fn -= 1;
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        let Pat::Ident(binding) = &local.pat else {
            visit::visit_local(self, local);
            return;
        };
        if binding.ident == "_gpu_lock" {
            self.error = true;
        }
        if binding.ident != "encode" {
            visit::visit_local(self, local);
            return;
        }
        let Some(helper) = self.helper() else {
            self.error = true;
            return;
        };
        let Some(init) = &local.init else {
            self.error = true;
            return;
        };
        let Expr::Closure(closure) = init.expr.as_ref() else {
            self.error = true;
            return;
        };
        let typed_encoder = match closure.inputs.first() {
            Some(Pat::Type(argument)) => {
                matches!(argument.pat.as_ref(), Pat::Ident(name)
                if name.ident == "encoder" && name.mutability.is_none() && name.by_ref.is_none()
                    && name.subpat.is_none())
                    && matches!(argument.ty.as_ref(), syn::Type::Reference(reference)
                        if reference.mutability.is_none() && matches!(reference.elem.as_ref(), syn::Type::Path(path)
                            if path.qself.is_none() && path.path.segments.len() == 2
                                && path.path.segments[0].ident == "metal"
                                && path.path.segments[1].ident == "ComputeCommandEncoderRef"))
            }
            _ => false,
        };
        if !typed_encoder
            || self.closure_depth != 0
            || self.nested_fn != 0
            || !local.attrs.is_empty()
            || binding.mutability.is_some()
            || binding.by_ref.is_some()
            || binding.subpat.is_some()
            || init.diverge.is_some()
            || !closure.attrs.is_empty()
            || closure.capture.is_some()
            || closure.asyncness.is_some()
            || closure.inputs.len() != 1
        {
            self.error = true;
        }
        self.closures[helper] += 1;
        self.closure_depth += 1;
        self.visit_expr(&closure.body);
        self.closure_depth -= 1;
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.closure_depth += 1;
        visit::visit_expr_closure(self, closure);
        self.closure_depth -= 1;
    }

    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        if pat.ident == "_gpu_lock" || pat.ident == "encode" {
            self.error = true;
        }
        visit::visit_pat_ident(self, pat);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        for part in &path.path.segments {
            let id = part.ident.to_string();
            if matches!(
                id.as_str(),
                "_gpu_lock" | "gpu_test_lock" | "run" | "timed_dispatch" | "encode"
            ) || HELPERS.contains(&id.as_str())
            {
                self.error = true;
            }
        }
        visit::visit_expr_path(self, path);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let name = if path_is(&call.func, &["run"]) {
            Some("run")
        } else if path_is(&call.func, &["timed_dispatch"]) {
            Some("timed_dispatch")
        } else if path_is(&call.func, &["encode"]) {
            Some("encode")
        } else {
            HELPERS
                .iter()
                .find(|name| path_is(&call.func, &[*name]))
                .copied()
        };
        let Some(name) = name else {
            visit::visit_expr_call(self, call);
            return;
        };
        if !call.attrs.is_empty() || self.nested_fn != 0 || self.closure_depth != 0 {
            self.error = true;
        }
        match name {
            "run" => {
                if self.current != "main" || !call.args.is_empty() {
                    self.error = true;
                }
                self.run_calls += 1;
            }
            "timed_dispatch" => {
                let Some(helper) = self.helper() else {
                    self.error = true;
                    return;
                };
                if call.args.len() != 2
                    || !path_is(&call.args[0], &["queue"])
                    || !path_is(&call.args[1], &["encode"])
                {
                    self.error = true;
                }
                self.timings[helper] += 1;
                // `encode` is admitted only as this direct synchronous argument.
                return;
            }
            "encode" => {
                if self.current != "timed_dispatch"
                    || call.args.len() != 1
                    || !path_is(&call.args[0], &["encoder"])
                {
                    self.error = true;
                }
                self.timed_work[2] += 1;
            }
            _ => {
                let helper = HELPERS.iter().position(|item| *item == name).unwrap();
                if self.current != "run" {
                    self.error = true;
                }
                self.helpers[helper] += 1;
            }
        }
        for arg in &call.args {
            self.visit_expr(arg);
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let (index, receiver) = match call.method.to_string().as_str() {
            "new_command_buffer" => (Some(0), "queue"),
            "new_compute_command_encoder" => (Some(1), "command"),
            "end_encoding" => (Some(3), "encoder"),
            "commit" => (Some(4), "command"),
            "wait_until_completed" => (Some(5), "command"),
            "dispatch_thread_groups" => {
                if let Some(helper) = self.helper() {
                    if self.closure_depth != 1
                        || self.nested_fn != 0
                        || !path_is(&call.receiver, &["encoder"])
                        || !call.attrs.is_empty()
                    {
                        self.error = true;
                    }
                    self.dispatches[helper] += 1;
                } else {
                    self.error = true;
                }
                (None, "")
            }
            _ => (None, ""),
        };
        if let Some(index) = index {
            if self.current != "timed_dispatch"
                || self.closure_depth != 0
                || self.nested_fn != 0
                || !path_is(&call.receiver, &[receiver])
                || !call.attrs.is_empty()
            {
                self.error = true;
            }
            self.timed_work[index] += 1;
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
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if metal_cfg(&function.attrs) => Some(function),
            _ => None,
        })
        .collect();
    let main: Vec<_> = functions.iter().filter(|f| f.sig.ident == "main").collect();
    let [main] = main.as_slice() else {
        return Err("Metal main");
    };
    if file
        .items
        .iter()
        .filter(|item| matches!(item, Item::Fn(function) if function.sig.ident == "main"))
        .count()
        != 2
    {
        return Err("main declarations");
    }
    if main.block.stmts.first().is_none_or(|stmt| !guard(stmt)) {
        return Err("guard");
    }
    for name in HELPERS.into_iter().chain(["run", "timed_dispatch"]) {
        if functions.iter().filter(|f| f.sig.ident == name).count() != 1 {
            return Err("helper declaration");
        }
    }
    let mut audit = Audit::default();
    for item in &file.items {
        if let Item::Fn(function) = item {
            let metal_main = function.sig.ident == "main" && metal_cfg(&function.attrs);
            audit.current = if function.sig.ident == "main" && !metal_main {
                "non_metal_main".to_string()
            } else {
                function.sig.ident.to_string()
            };
            let stmts = &function.block.stmts;
            let start = usize::from(metal_main);
            for stmt in &stmts[start..] {
                audit.visit_stmt(stmt);
            }
        } else {
            audit.current.clear();
            match item {
                Item::Const(item) => audit.visit_expr(&item.expr),
                Item::Static(item) => audit.visit_expr(&item.expr),
                Item::Impl(item) => {
                    for member in &item.items {
                        audit.visit_impl_item(member);
                    }
                }
                Item::Macro(item) => audit.visit_macro(&item.mac),
                _ => {}
            }
        }
    }
    if audit.error
        || audit.run_calls != 1
        || audit.helpers != [1, 1, 1, 1, 1]
        || audit.closures != [1, 1, 1, 1, 1]
        || audit.timings != [1, 2, 2, 2, 2]
        || audit.dispatches != [1, 1, 1, 1, 1]
        || audit.timed_work != [1, 1, 1, 1, 1, 1]
    {
        return Err("work chain");
    }
    Ok(())
}

#[test]
fn gemm_shapes_dispatch_chain_keeps_the_guard() {
    assert_eq!(check(SOURCE), Ok(()));
}

#[test]
fn gemm_shapes_contract_rejects_source_mutations() {
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
            "clock_warmup(&device, &queue)?;",
            "let warmup = clock_warmup; warmup(&device, &queue)?;",
        ),
        (
            "let encode = |encoder: &metal::ComputeCommandEncoderRef|",
            "let encode = move |encoder: &metal::ComputeCommandEncoderRef|",
        ),
        (
            "let encode = |encoder: &metal::ComputeCommandEncoderRef|",
            "let encode = |other: &metal::ComputeCommandEncoderRef|",
        ),
        (
            "timed_dispatch(queue, encode)?;",
            "let deferred = || timed_dispatch(queue, encode); deferred()?;",
        ),
        ("encode(encoder);", "let saved = encode; saved(encoder);"),
        ("command.wait_until_completed();", "command.commit();"),
        (
            "clock_warmup(&device, &queue)?;",
            "#[cfg(any())] clock_warmup(&device, &queue)?;",
        ),
        (
            "timed_dispatch(queue, encode)?;",
            "macro_rules! hidden { () => { timed_dispatch(queue, encode) } } hidden!()?;",
        ),
    ];
    for (old, new) in mutations {
        assert!(SOURCE.contains(old), "missing mutation anchor: {old}");
        assert!(
            check(&SOURCE.replacen(old, new, 1)).is_err(),
            "accepted mutation: {old}"
        );
    }
}
