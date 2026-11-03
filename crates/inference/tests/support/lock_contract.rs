use syn::visit::{self, Visit};
use syn::{Attribute, Expr, Meta, Pat, Stmt};

// Each integration test compiles this module separately and uses only its own actions.
#[allow(dead_code)]
pub(crate) enum MacroAction {
    Skip,
    Walk,
    Expressions(Vec<Expr>),
}

pub(crate) trait WorkPolicy {
    fn deferred(&mut self) -> &mut usize;
    fn attribute(&mut self, _attr: &syn::Attribute) {}
    fn pat_ident(&mut self, _pat: &syn::PatIdent) {}
    fn expr_path(&mut self, _path: &syn::ExprPath) {}
    fn expr_call(&mut self, _call: &syn::ExprCall) {}
    fn expr_method_call(&mut self, _call: &syn::ExprMethodCall) {}
    fn local(&mut self, _local: &syn::Local) {}
    fn expr_assign(&mut self, _assign: &syn::ExprAssign) -> bool {
        true
    }
    fn macro_action(&mut self, mac: &syn::Macro) -> MacroAction;
}

pub(crate) struct Work<P> {
    policy: P,
}

impl<P> Work<P> {
    pub(crate) fn new(policy: P) -> Self {
        Self { policy }
    }
}

impl<P: Default> Default for Work<P> {
    fn default() -> Self {
        Self::new(P::default())
    }
}

impl<P> std::ops::Deref for Work<P> {
    type Target = P;
    fn deref(&self) -> &P {
        &self.policy
    }
}

impl<P> std::ops::DerefMut for Work<P> {
    fn deref_mut(&mut self) -> &mut P {
        &mut self.policy
    }
}

impl<'ast, P: WorkPolicy> Visit<'ast> for Work<P> {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.policy.attribute(attr);
        visit::visit_attribute(self, attr);
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.policy.pat_ident(pat);
        visit::visit_pat_ident(self, pat);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.policy.expr_path(path);
        visit::visit_expr_path(self, path);
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.policy.expr_call(call);
        visit::visit_expr_call(self, call);
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.policy.expr_method_call(call);
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        if self.policy.expr_assign(assign) {
            visit::visit_expr_assign(self, assign);
        }
    }
    fn visit_local(&mut self, local: &'ast syn::Local) {
        self.policy.local(local);
        visit::visit_local(self, local);
    }
    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        *self.policy.deferred() += 1;
        visit::visit_expr_closure(self, expr);
        *self.policy.deferred() -= 1;
    }
    fn visit_expr_async(&mut self, expr: &'ast syn::ExprAsync) {
        *self.policy.deferred() += 1;
        visit::visit_expr_async(self, expr);
        *self.policy.deferred() -= 1;
    }
    fn visit_item(&mut self, item: &'ast syn::Item) {
        *self.policy.deferred() += 1;
        visit::visit_item(self, item);
        *self.policy.deferred() -= 1;
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        match self.policy.macro_action(mac) {
            MacroAction::Skip => {}
            MacroAction::Walk => visit::visit_macro(self, mac),
            MacroAction::Expressions(expressions) => {
                for expr in &expressions {
                    self.visit_expr(expr);
                }
            }
        }
    }
}

// The suite and direct-metal contracts audit individual functions instead.
#[allow(dead_code)]
pub(crate) fn visit_file_bodies<P: WorkPolicy>(file: &syn::File, mut work: Work<P>) -> Work<P> {
    for item in &file.items {
        if let syn::Item::Fn(function) = item {
            work.visit_block(&function.block);
        } else {
            work.visit_item(item);
        }
    }
    work
}

pub(crate) fn check_counts(counts: &[(&str, usize, usize)]) -> Result<(), String> {
    for &(name, observed, expected) in counts {
        if observed != expected {
            return Err(format!("{name}: observed {observed}, expected {expected}"));
        }
    }
    Ok(())
}

// The prefill/pruning contracts keep their own mutation set.
#[allow(dead_code)]
pub(crate) fn assert_guard_mutations(
    source: &str,
    anchor: &str,
    insert_after: &str,
    guard: &str,
    expected: [&str; 3],
    check: impl Fn(&str) -> Result<(), String>,
) {
    assert_eq!(
        source.matches(anchor).count(),
        1,
        "shared guard mutation anchor"
    );
    assert_eq!(
        anchor.matches(insert_after).count(),
        1,
        "guard insertion point"
    );
    for ((name, insertion), expected) in [
        ("drop guard", format!("drop({guard});")),
        ("shadow guard", format!("let {guard} = ();")),
        ("opaque macro", "hidden_work!();".to_owned()),
    ]
    .into_iter()
    .zip(expected)
    {
        let replacement = anchor.replacen(
            insert_after,
            &format!("{insert_after}\n    {insertion}\n"),
            1,
        );
        let mutant = source.replacen(anchor, &replacement, 1);
        assert!(syn::parse_file(&mutant).is_ok(), "{name} must parse");
        assert_eq!(check(&mutant), Err(expected.into()), "{name}");
    }
}

fn trusted_macro(name: &str) -> bool {
    matches!(
        name,
        "println"
            | "eprintln"
            | "format"
            | "vec"
            | "matches"
            | "assert"
            | "assert_eq"
            | "panic"
            | "thread_local"
    )
}

fn opaque_tokens(tokens: proc_macro2::TokenStream) -> bool {
    use proc_macro2::TokenTree;
    let tokens: Vec<_> = tokens.into_iter().collect();
    for (index, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Group(group) if opaque_tokens(group.stream()) => return true,
            TokenTree::Ident(name) if matches!(tokens.get(index + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!') => {
                if !trusted_macro(&name.to_string())
                    || matches!(index.checked_sub(1).and_then(|i| tokens.get(i)),
                        Some(TokenTree::Punct(p)) if p.as_char() == ':')
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn shadowing_import(tree: &syn::UseTree) -> bool {
    match tree {
        syn::UseTree::Path(path) => shadowing_import(&path.tree),
        syn::UseTree::Name(name) => trusted_macro(&name.ident.to_string()),
        syn::UseTree::Rename(rename) => trusted_macro(&rename.rename.to_string()),
        syn::UseTree::Group(group) => group.items.iter().any(shadowing_import),
        syn::UseTree::Glob(_) => true,
    }
}

fn contains_macro_use(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(ident) => ident == "macro_use",
        proc_macro2::TokenTree::Group(group) => contains_macro_use(group.stream()),
        _ => false,
    })
}

#[derive(Default)]
struct MacroAudit {
    bad: bool,
    test_depth: usize,
}

impl<'ast> Visit<'ast> for MacroAudit {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let is_test_module = item.attrs.iter().any(|attr| {
            attr.path().is_ident("cfg")
                && matches!(attr.parse_args::<Meta>(), Ok(Meta::Path(path)) if path.is_ident("test"))
        });
        if is_test_module {
            self.test_depth += 1;
        }
        visit::visit_item_mod(self, item);
        if is_test_module {
            self.test_depth -= 1;
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if self.test_depth == 0 {
            self.bad |= shadowing_import(&item.tree);
        }
        visit::visit_item_use(self, item);
    }

    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        self.bad |= attr.path().is_ident("macro_use")
            || (attr.path().is_ident("cfg_attr")
                && matches!(&attr.meta, Meta::List(list) if contains_macro_use(list.tokens.clone())));
        visit::visit_attribute(self, attr);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.bad |= mac
            .path
            .get_ident()
            .is_none_or(|name| !trusted_macro(&name.to_string()))
            || opaque_tokens(mac.tokens.clone());
    }
}

pub(crate) fn macro_policy(file: &syn::File) -> Result<(), &'static str> {
    let mut audit = MacroAudit::default();
    audit.visit_file(file);
    if audit.bad {
        Err("opaque or shadowed macro")
    } else {
        Ok(())
    }
}

pub(crate) fn check(source: &str) -> Result<syn::File, String> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse".to_owned())?;
    macro_policy(&file)?;
    Ok(file)
}

pub(crate) fn captures_format(value: &str, names: &[&str]) -> bool {
    value.split('{').skip(1).any(|suffix| {
        let name: String = suffix
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
        names.contains(&name.as_str())
    })
}

pub(crate) fn sensitive_tokens(
    tokens: proc_macro2::TokenStream,
    names: &[&str],
    captures: &[&str],
) -> bool {
    tokens.into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => names.iter().any(|name| id == *name),
        proc_macro2::TokenTree::Group(group) => sensitive_tokens(group.stream(), names, captures),
        proc_macro2::TokenTree::Literal(literal) => {
            syn::parse_str::<syn::LitStr>(&literal.to_string())
                .is_ok_and(|value| captures_format(&value.value(), captures))
        }
        _ => false,
    })
}

pub(crate) fn path_is(expr: &Expr, names: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == names.len()
        && path.path.segments.iter().zip(names).all(|(part, name)| {
            part.ident == *name && matches!(part.arguments, syn::PathArguments::None)
        })
}

pub(crate) fn metal_cfg(attrs: &[Attribute], counters: bool) -> bool {
    let mut attrs = attrs.iter().filter(|attr| !attr.path().is_ident("doc"));
    let (Some(attr), None) = (attrs.next(), attrs.next()) else {
        return false;
    };
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
    args.len() == (if counters { 3 } else { 2 })
        && pair(&args[0], "target_os", "macos")
        && pair(&args[1], "feature", "metal-gpu")
        && (!counters || pair(&args[2], "feature", "gdn-state-counters"))
}

pub(crate) fn guard(stmt: &Stmt, name: &str, conditional: bool) -> bool {
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
    (if conditional {
        metal_cfg(&local.attrs, false)
    } else {
        local.attrs.is_empty()
    }) && binding.ident == name
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

#[cfg(test)]
mod tests {
    use super::{
        MacroAction, Work, WorkPolicy, check_counts, guard, macro_policy, metal_cfg,
        sensitive_tokens,
    };
    use syn::visit::Visit;
    use syn::{Attribute, Stmt, parse_quote};

    #[test]
    fn cfg_accepts_both_metal_forms_and_inert_docs() {
        let two: Vec<Attribute> = vec![
            parse_quote!(#[doc = "fixture"]),
            parse_quote!(#[cfg(all(target_os = "macos", feature = "metal-gpu"))]),
        ];
        let three: Vec<Attribute> = vec![parse_quote!(#[cfg(all(
            target_os = "macos",
            feature = "metal-gpu",
            feature = "gdn-state-counters"
        ))])];
        assert!(metal_cfg(&two, false));
        assert!(!metal_cfg(&two, true));
        assert!(metal_cfg(&three, true));
        assert!(!metal_cfg(&three, false));
    }

    #[test]
    fn guard_preserves_named_and_conditional_variants() {
        let direct: Stmt =
            parse_quote!(let _gpu_guard = lattice_inference::measurement::gpu_test_lock(););
        let conditional: Stmt = parse_quote!(
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            let _gpu_lock = lattice_inference::measurement::gpu_test_lock();
        );
        assert!(guard(&direct, "_gpu_guard", false));
        assert!(!guard(&direct, "_gpu_lock", false));
        assert!(guard(&conditional, "_gpu_lock", true));
        assert!(!guard(&conditional, "_gpu_lock", false));
    }

    #[test]
    fn opaque_and_shadowed_macro_fixtures_are_rejected() {
        let accepted = syn::parse_file("fn main() { println!(\"ok\"); }").unwrap();
        assert_eq!(macro_policy(&accepted), Ok(()));
        let test_import = syn::parse_file("#[cfg(test)] mod tests { use super::*; }").unwrap();
        assert_eq!(macro_policy(&test_import), Ok(()));
        for (name, source) in [
            ("opaque invocation", "fn main() { hidden_work!(); }"),
            (
                "nested invocation",
                "fn main() { println!(\"{}\", hidden_work!()); }",
            ),
            (
                "shadowing definition",
                "macro_rules! println { () => {} } fn main() {}",
            ),
            (
                "shadowing import",
                "use other::hidden as println; fn main() {}",
            ),
            (
                "conditional macro import",
                "#[cfg_attr(feature = \"x\", macro_use)] mod imports {} fn main() {}",
            ),
            (
                "test module macro definition",
                "#[cfg(test)] mod tests { macro_rules! println { () => {} } }",
            ),
            ("glob import", "use other::*; fn main() {}"),
        ] {
            let file = syn::parse_file(source).unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_eq!(
                macro_policy(&file),
                Err("opaque or shadowed macro"),
                "{name}"
            );
        }
    }

    #[test]
    fn sensitive_tokens_match_nested_names_and_exact_format_captures() {
        let names = &["state", "reset_state"];
        let captures = &["state"];
        for (source, expected) in [
            ("nested(state.reset_state())", true),
            (r#""{state:?}""#, true),
            (r#""{state_bytes_moved}""#, false),
            ("nested(other)", false),
        ] {
            let tokens = source.parse().unwrap();
            assert_eq!(
                sensitive_tokens(tokens, names, captures),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn common_count_diagnostic_names_the_first_mismatch() {
        assert_eq!(check_counts(&[("work", 2, 2)]), Ok(()));
        assert_eq!(
            check_counts(&[("work", 2, 3)]),
            Err("work: observed 2, expected 3".into())
        );
    }

    #[test]
    fn deferred_visitor_restores_depth_after_nested_work() {
        #[derive(Default)]
        struct Probe {
            deferred: usize,
            depths: Vec<usize>,
        }
        impl WorkPolicy for Probe {
            fn deferred(&mut self) -> &mut usize {
                &mut self.deferred
            }
            fn expr_path(&mut self, path: &syn::ExprPath) {
                if path.path.is_ident("target") {
                    self.depths.push(self.deferred);
                }
            }
            fn macro_action(&mut self, _mac: &syn::Macro) -> MacroAction {
                MacroAction::Walk
            }
        }
        let expr: syn::Expr = parse_quote!(|| async { target });
        let mut probe = Work::new(Probe::default());
        probe.visit_expr(&expr);
        assert_eq!(probe.depths, [2]);
        assert_eq!(probe.deferred, 0);
    }

    #[test]
    fn policy_controls_assignment_and_macro_expression_traversal() {
        #[derive(Default)]
        struct Probe {
            deferred: usize,
            calls: usize,
        }
        impl WorkPolicy for Probe {
            fn deferred(&mut self) -> &mut usize {
                &mut self.deferred
            }
            fn expr_assign(&mut self, _assign: &syn::ExprAssign) -> bool {
                false
            }
            fn expr_call(&mut self, _call: &syn::ExprCall) {
                self.calls += 1;
            }
            fn macro_action(&mut self, _mac: &syn::Macro) -> MacroAction {
                MacroAction::Expressions(vec![parse_quote!(target())])
            }
        }
        let mut probe = Work::new(Probe::default());
        let invocation: syn::Expr = parse_quote!(example!());
        probe.visit_expr(&invocation);
        assert_eq!(probe.calls, 1);
        let assignment: syn::Expr = parse_quote!(left = target());
        probe.visit_expr(&assignment);
        assert_eq!(probe.calls, 1);
    }
}
