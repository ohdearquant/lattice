use syn::parse::{Parse, ParseStream, Parser};
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
    fn item(&mut self, _item: &syn::Item) {}
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
        self.policy.item(item);
        *self.policy.deferred() += 1;
        visit::visit_item(self, item);
        *self.policy.deferred() -= 1;
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        match self.policy.macro_action(mac) {
            MacroAction::Skip | MacroAction::Walk => match macro_arguments(mac) {
                Ok(MacroArguments::Expressions(expressions)) => {
                    for expr in &expressions {
                        if !matches!(expr, Expr::Lit(_)) {
                            self.visit_expr(expr);
                        }
                    }
                }
                Ok(MacroArguments::Matches(arguments)) => {
                    self.visit_expr(&arguments.expression);
                    self.visit_pat(&arguments.pattern);
                    if let Some(guard) = &arguments.guard {
                        self.visit_expr(guard);
                    }
                }
                Ok(MacroArguments::Items(file)) => {
                    for item in &file.items {
                        self.visit_item(item);
                    }
                }
                Err(()) => {} // The file-level audit rejects this before work is visited.
            },
            MacroAction::Expressions(expressions) => {
                for expr in &expressions {
                    if !matches!(expr, Expr::Lit(_)) {
                        self.visit_expr(expr);
                    }
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

#[allow(dead_code)]
pub(crate) fn hidden_gpu_work_mutant(source: &str, entry: &str, guard: &str) -> String {
    const DEFINITION: &str = "macro_rules! hidden_gpu_work { () => {{ let _hidden_guard = lattice_inference::measurement::gpu_test_lock(); }}; }\n";
    let entry_index = source.find(entry).expect("missing entry mutation anchor");
    let cfg_index = source[..entry_index]
        .rfind("#[cfg(")
        .expect("missing entry cfg mutation anchor");
    assert!(source.contains(guard), "missing guard mutation anchor");
    let mut mutated = source.to_owned();
    mutated.insert_str(cfg_index, DEFINITION);
    let mutated = mutated.replacen(guard, &format!("{guard}\n    hidden_gpu_work!();\n"), 1);
    assert!(
        syn::parse_file(&mutated).is_ok(),
        "hidden macro mutant must parse"
    );
    mutated
}

#[allow(dead_code)]
pub(crate) fn mutate_run_cfg(source: &str) -> String {
    const CFG: &str = "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]";
    assert_eq!(source.matches("fn run(").count(), 1, "run mutation anchor");
    let run = source.find("fn run(").expect("run mutation anchor");
    let cfg = source[..run].rfind(CFG).expect("run Metal cfg anchor");
    let prefix = &source[cfg..run];
    let fixture: syn::ItemFn =
        syn::parse_str(&format!("{prefix}fn run() {{}}")).expect("cfg must directly attach to run");
    assert!(metal_cfg(&fixture.attrs, false), "original run cfg");
    let mut mutated = source.to_owned();
    mutated.replace_range(cfg..cfg + CFG.len(), "#[cfg(any())]");
    let mutated_prefix = &mutated[cfg..run - CFG.len() + "#[cfg(any())]".len()];
    let fixture: syn::ItemFn = syn::parse_str(&format!("{mutated_prefix}fn run() {{}}"))
        .expect("mutated cfg must directly attach to run");
    assert!(!metal_cfg(&fixture.attrs, false), "mutated run cfg");
    mutated
}

#[allow(dead_code)]
pub(crate) struct GuardVector<'a> {
    pub(crate) expression: &'a str,
    pub(crate) expected: &'a str,
}

// The prefill/pruning contracts keep their own mutation set.
#[allow(dead_code)]
pub(crate) fn assert_guard_mutations(
    source: &str,
    anchor: &str,
    insert_after: &str,
    guard: &str,
    expected: [&str; 3],
    vector: GuardVector<'_>,
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
    let before =
        |insertion: &str| source.replacen(anchor, &format!("    {insertion}\n{anchor}"), 1);
    let after = |insertion: &str| {
        let replacement = anchor.replacen(
            insert_after,
            &format!("{insert_after}\n    {insertion}\n"),
            1,
        );
        source.replacen(anchor, &replacement, 1)
    };
    for (name, mutated, expected) in [
        (
            "pre-guard vec work",
            before(&format!("let _v = vec![{}];", vector.expression)),
            vector.expected,
        ),
        ("drop guard", after(&format!("drop({guard});")), expected[0]),
        (
            "shadow guard",
            after(&format!("let {guard} = ();")),
            expected[1],
        ),
        ("opaque macro", after("hidden_work!();"), expected[2]),
    ] {
        assert!(syn::parse_file(&mutated).is_ok(), "{name} must parse");
        assert_eq!(check(&mutated), Err(expected.into()), "{name}");
    }
}

fn trusted_macro(name: &str) -> bool {
    matches!(
        name,
        "print"
            | "println"
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

fn trusted_macro_path(path: &syn::Path) -> bool {
    let mut segments = path.segments.iter();
    let Some(first) = segments.next() else {
        return false;
    };
    if !first.arguments.is_empty() {
        return false;
    }
    match (segments.next(), segments.next()) {
        (None, None) => path.leading_colon.is_none() && trusted_macro(&first.ident.to_string()),
        (Some(name), None) => {
            (first.ident == "std" || first.ident == "core")
                && name.arguments.is_empty()
                && trusted_macro(&name.ident.to_string())
        }
        _ => false,
    }
}

enum MacroArguments {
    Expressions(Vec<Expr>),
    Matches(Box<MatchesArguments>),
    Items(Box<syn::File>),
}

struct MatchesArguments {
    expression: Expr,
    pattern: Pat,
    guard: Option<Expr>,
}

impl Parse for MatchesArguments {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let expression = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let pattern = Pat::parse_multi_with_leading_vert(input)?;
        let guard = if input.peek(syn::Token![if]) {
            input.parse::<syn::Token![if]>()?;
            Some(input.parse()?)
        } else {
            None
        };
        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
        Ok(Self {
            expression,
            pattern,
            guard,
        })
    }
}

fn macro_arguments(mac: &syn::Macro) -> Result<MacroArguments, ()> {
    let name = mac.path.segments.last().ok_or(())?.ident.to_string();
    if name == "thread_local" {
        return syn::parse2(mac.tokens.clone())
            .map(|file| MacroArguments::Items(Box::new(file)))
            .map_err(|_| ());
    }
    if name == "matches" {
        return syn::parse2(mac.tokens.clone())
            .map(|args| MacroArguments::Matches(Box::new(args)))
            .map_err(|_| ());
    }
    let expressions = syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated
        .parse2(mac.tokens.clone());
    if let Ok(expressions) = expressions {
        return Ok(MacroArguments::Expressions(
            expressions.into_iter().collect(),
        ));
    }
    if name == "vec" {
        let group = proc_macro2::Group::new(proc_macro2::Delimiter::Bracket, mac.tokens.clone());
        let tokens = proc_macro2::TokenStream::from(proc_macro2::TokenTree::Group(group));
        if let Ok(expression @ Expr::Repeat(_)) = syn::parse2(tokens) {
            return Ok(MacroArguments::Expressions(vec![expression]));
        }
    }
    Err(())
}

fn protected_root(name: &str) -> bool {
    matches!(
        name.trim_start_matches("r#"),
        "std" | "core" | "lattice_inference"
    )
}

fn opaque_tokens(tokens: proc_macro2::TokenStream) -> bool {
    use proc_macro2::TokenTree;
    fn colon(token: Option<&TokenTree>) -> bool {
        matches!(token, Some(TokenTree::Punct(p)) if p.as_char() == ':')
    }

    let tokens: Vec<_> = tokens.into_iter().collect();
    for (index, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Group(group) if opaque_tokens(group.stream()) => return true,
            TokenTree::Ident(name)
                if matches!(tokens.get(index + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!')
                    && matches!(tokens.get(index + 2), Some(TokenTree::Group(_))) =>
            {
                let qualified =
                    index >= 2 && colon(tokens.get(index - 1)) && colon(tokens.get(index - 2));
                let trusted_qualifier = if qualified && index >= 3 {
                    let root = index - 3;
                    let standard = matches!(tokens.get(root), Some(TokenTree::Ident(id))
                        if id == "std" || id == "core");
                    let earlier_path =
                        root >= 2 && colon(tokens.get(root - 1)) && colon(tokens.get(root - 2));
                    standard && (!earlier_path || root == 2)
                } else {
                    !qualified && !index.checked_sub(1).is_some_and(|i| colon(tokens.get(i)))
                };
                if !trusted_macro(&name.to_string()) || !trusted_qualifier {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn shadowing_import(tree: &syn::UseTree, prefix: &mut Vec<String>) -> bool {
    let binding = |name: &syn::Ident, rename: Option<&syn::Ident>| {
        let mut origin = prefix.clone();
        if name != "self" {
            origin.push(name.to_string().trim_start_matches("r#").to_owned());
        }
        let bound = rename
            .map(|name| name.to_string().trim_start_matches("r#").to_owned())
            .or_else(|| origin.last().cloned())
            .unwrap_or_default();
        if protected_root(&bound) {
            return origin.len() != 1 || origin[0] != bound;
        }
        if !trusted_macro(&bound) {
            return false;
        }
        origin.len() != 2
            || origin[1] != bound
            || !(origin[0] == "std"
                || (origin[0] == "core"
                    && matches!(bound.as_str(), "matches" | "assert" | "assert_eq" | "panic")))
    };
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string().trim_start_matches("r#").to_owned());
            let shadowed = shadowing_import(&path.tree, prefix);
            prefix.pop();
            shadowed
        }
        syn::UseTree::Name(name) => binding(&name.ident, None),
        syn::UseTree::Rename(rename) => binding(&rename.ident, Some(&rename.rename)),
        syn::UseTree::Group(group) => group
            .items
            .iter()
            .any(|tree| shadowing_import(tree, prefix)),
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
    bad_arguments: bool,
    test_depth: usize,
}

impl<'ast> Visit<'ast> for MacroAudit {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        self.bad |= protected_root(&item.ident.to_string());
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

    fn visit_item_extern_crate(&mut self, item: &'ast syn::ItemExternCrate) {
        if let Some((_, alias)) = &item.rename {
            self.bad |= protected_root(&alias.to_string()) && alias != &item.ident;
        }
        visit::visit_item_extern_crate(self, item);
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        let test_super_glob = self.test_depth > 0
            && matches!(&item.tree, syn::UseTree::Path(path)
                if path.ident == "super" && matches!(path.tree.as_ref(), syn::UseTree::Glob(_)));
        if !test_super_glob {
            self.bad |= shadowing_import(&item.tree, &mut Vec::new());
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
        if !trusted_macro_path(&mac.path) || opaque_tokens(mac.tokens.clone()) {
            self.bad = true;
            return;
        }
        match macro_arguments(mac) {
            Ok(MacroArguments::Expressions(expressions)) => {
                for expr in &expressions {
                    self.visit_expr(expr);
                }
            }
            Ok(MacroArguments::Matches(arguments)) => {
                self.visit_expr(&arguments.expression);
                self.visit_pat(&arguments.pattern);
                if let Some(guard) = &arguments.guard {
                    self.visit_expr(guard);
                }
            }
            Ok(MacroArguments::Items(file)) => self.visit_file(&file),
            Err(()) => self.bad_arguments = true,
        }
    }
}

pub(crate) fn macro_policy(file: &syn::File) -> Result<(), &'static str> {
    let mut audit = MacroAudit::default();
    audit.visit_file(file);
    if audit.bad {
        Err("opaque or shadowed macro")
    } else if audit.bad_arguments {
        Err("unparseable trusted macro arguments")
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
    use syn::parse::Parser;
    use syn::visit::Visit;
    use syn::{Attribute, Stmt};

    #[test]
    fn run_cfg_mutation_preserves_earlier_main_entry_cfg() {
        let source = r#"
            fn main() {
                #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
                run();
            }
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            fn run() {}
        "#;
        let mutated = super::mutate_run_cfg(source);
        let file = syn::parse_file(&mutated).expect("mutant must parse");
        let [syn::Item::Fn(main), syn::Item::Fn(run)] = file.items.as_slice() else {
            panic!("fixture needs main and run");
        };
        let [Stmt::Expr(syn::Expr::Call(call), _)] = main.block.stmts.as_slice() else {
            panic!("fixture needs direct entry call");
        };
        assert!(
            metal_cfg(&call.attrs, false),
            "main entry cfg remains Metal"
        );
        assert!(!metal_cfg(&run.attrs, false), "run cfg is changed");
    }

    #[test]
    fn cfg_accepts_both_metal_forms_and_inert_docs() {
        let two: Vec<Attribute> = Attribute::parse_outer
            .parse_str(
                r#"#[doc = "fixture"] #[cfg(all(target_os = "macos", feature = "metal-gpu"))]"#,
            )
            .expect("two fixture attributes");
        let three: Vec<Attribute> = Attribute::parse_outer
            .parse_str(r#"#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "gdn-state-counters"))]"#)
            .expect("three-feature fixture attribute");
        assert!(metal_cfg(&two, false));
        assert!(!metal_cfg(&two, true));
        assert!(metal_cfg(&three, true));
        assert!(!metal_cfg(&three, false));
    }

    #[test]
    fn guard_preserves_named_and_conditional_variants() {
        let direct: Stmt =
            syn::parse_str("let _gpu_guard = lattice_inference::measurement::gpu_test_lock();")
                .expect("direct guard fixture");
        let conditional: Stmt = syn::parse_str(
            r#"#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            let _gpu_lock = lattice_inference::measurement::gpu_test_lock();"#,
        )
        .expect("conditional guard fixture");
        assert!(guard(&direct, "_gpu_guard", false));
        assert!(!guard(&direct, "_gpu_lock", false));
        assert!(guard(&conditional, "_gpu_lock", true));
        assert!(!guard(&conditional, "_gpu_lock", false));
    }

    #[test]
    fn opaque_and_shadowed_macro_fixtures_are_rejected() {
        let accepted = syn::parse_file("fn main() { println!(\"ok\"); }").unwrap();
        assert_eq!(macro_policy(&accepted), Ok(()));
        let qualified = syn::parse_file(
            "std::thread_local! { static SLOT: u8 = 0; } fn main() { core::assert!(true); ::std::println!(\"{:?}\", std::vec![1]); }",
        )
        .unwrap();
        assert_eq!(macro_policy(&qualified), Ok(()));
        let operators = syn::parse_file("fn main() { assert!(a != b); assert!(!x); }").unwrap();
        assert_eq!(macro_policy(&operators), Ok(()));
        let non_list = syn::parse_file(
            "fn main() { vec![value; count]; matches!(x, Some(y) | None if predicate()); }",
        )
        .unwrap();
        assert_eq!(macro_policy(&non_list), Ok(()));
        let invalid_args = syn::parse_file("fn main() { vec![value; count; extra]; }").unwrap();
        assert_eq!(
            macro_policy(&invalid_args),
            Err("unparseable trusted macro arguments")
        );
        let test_import = syn::parse_file("#[cfg(test)] mod tests { use super::*; }").unwrap();
        assert_eq!(macro_policy(&test_import), Ok(()));
        for (name, source) in [
            ("opaque invocation", "fn main() { hidden_work!(); }"),
            ("custom print", "fn main() { other::print!(\"ok\"); }"),
            (
                "opaque print argument",
                "fn main() { print!(\"{}\", hidden_work!()); }",
            ),
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
                "nonstandard qualified macro",
                "fn main() { other::println!(\"ok\"); }",
            ),
            (
                "nested nonstandard qualified macro",
                "fn main() { println!(\"{:?}\", other::vec![1]); }",
            ),
            (
                "shadowed standard module",
                "mod std {} fn main() { std::println!(\"ok\"); }",
            ),
            (
                "shadowed standard import",
                "use other as std; fn main() { std::println!(\"ok\"); }",
            ),
            (
                "shadowed core extern crate",
                "extern crate other as core; fn main() { core::assert!(true); }",
            ),
            (
                "test module external glob",
                "#[cfg(test)] mod tests { use other::*; }",
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
    fn imports_preserve_canonical_origins_and_macro_identities() {
        for source in [
            "fn main() { print!(\"ok\"); }",
            "fn main() { std::print!(\"ok\"); }",
            "fn main() { ::std::print!(\"ok\"); }",
            "use std::print; fn main() { print!(\"ok\"); }",
            "use std::{print as print}; fn main() { print!(\"ok\"); }",
            "use std::vec; fn main() { vec![1]; }",
            "use ::std::vec; fn main() { vec![1]; }",
            "use std::{self, vec, println}; fn main() { println!(\"ok\"); vec![1]; }",
            "use std::vec as vec; fn main() { vec![1]; }",
            "use {std::vec, core::{assert, matches}}; fn main() { assert!(true); }",
            "use ::core::{assert_eq, panic}; fn main() { assert_eq!(1, 1); }",
            "use lattice_inference::{self, measurement::gpu_test_lock}; fn main() {}",
            "extern crate lattice_inference; fn main() {}",
        ] {
            let file = syn::parse_file(source).expect("canonical import fixture must parse");
            assert_eq!(macro_policy(&file), Ok(()), "{source}");
        }
        for source in [
            "use other::print;",
            "use other::{print};",
            "use std::println as print;",
            "use std::{print as println};",
            "use core::print;",
            "use other::vec;",
            "use crate::std::vec;",
            "use self::std::vec;",
            "use std::other::vec;",
            "use std::format as vec;",
            "use std::{vec as println};",
            "use {std::vec, other::{hidden as println}};",
            "use other::{self as vec};",
            "use core::vec;",
        ] {
            let file = syn::parse_file(source).expect("shadowing import fixture must parse");
            assert_eq!(
                macro_policy(&file),
                Err("opaque or shadowed macro"),
                "{source}"
            );
        }
    }

    #[test]
    fn guard_root_cannot_be_rebound_by_modules_or_imports() {
        const GUARD: &str =
            "fn main() { let _guard = lattice_inference::measurement::gpu_test_lock(); }";
        assert_eq!(macro_policy(&syn::parse_file(GUARD).unwrap()), Ok(()));
        for declaration in [
            "mod lattice_inference { pub mod measurement { pub fn unlocked() {} pub use self::unlocked as gpu_test_lock; } }",
            "mod r#lattice_inference {}",
            "use other as lattice_inference;",
            "use other::lattice_inference;",
            "use other::{self as lattice_inference};",
            "use other::{nested::{hidden as lattice_inference}};",
            "use std::{vec, self as lattice_inference};",
            "extern crate other as lattice_inference;",
        ] {
            let file = syn::parse_file(&format!("{declaration}\n{GUARD}"))
                .expect("guard root fixture must parse");
            assert_eq!(
                macro_policy(&file),
                Err("opaque or shadowed macro"),
                "{declaration}"
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
        let expr: syn::Expr = syn::parse_str("|| async { target }").expect("deferred fixture");
        let mut probe = Work::new(Probe::default());
        probe.visit_expr(&expr);
        assert_eq!(probe.depths, [2]);
        assert_eq!(probe.deferred, 0);
        let block: syn::Block = syn::parse_str("{ fn nested() { target; } target; }")
            .expect("default item policy fixture");
        probe.visit_block(&block);
        assert_eq!(probe.depths, [2, 1, 0]);
        assert_eq!(probe.deferred, 0);
    }

    #[test]
    fn item_policy_observes_each_item_without_skipping_nested_work() {
        #[derive(Default)]
        struct Probe {
            deferred: usize,
            items: Vec<(String, usize)>,
            paths: Vec<usize>,
        }
        impl WorkPolicy for Probe {
            fn deferred(&mut self) -> &mut usize {
                &mut self.deferred
            }
            fn item(&mut self, item: &syn::Item) {
                let name = match item {
                    syn::Item::Mod(item) => &item.ident,
                    syn::Item::Fn(item) => &item.sig.ident,
                    _ => panic!("unexpected fixture item"),
                };
                self.items.push((name.to_string(), self.deferred));
            }
            fn expr_path(&mut self, path: &syn::ExprPath) {
                if path.path.is_ident("target") {
                    self.paths.push(self.deferred);
                }
            }
            fn macro_action(&mut self, _mac: &syn::Macro) -> MacroAction {
                MacroAction::Walk
            }
        }
        let file = syn::parse_file(
            "mod outer { fn first() { target; fn nested() { target; } target; } }
             fn last() { target; }",
        )
        .expect("nested item fixture");
        let mut probe = Work::new(Probe::default());
        probe.visit_file(&file);
        assert_eq!(
            probe.items,
            [("outer", 0), ("first", 1), ("nested", 2), ("last", 0)]
                .map(|(name, depth)| (name.to_owned(), depth))
        );
        assert_eq!(probe.paths, [2, 3, 2, 1]);
        assert_eq!(probe.deferred, 0);
    }

    #[test]
    fn trusted_vec_arguments_reach_the_work_visitor() {
        #[derive(Default)]
        struct Probe {
            deferred: usize,
            calls: usize,
        }
        impl WorkPolicy for Probe {
            fn deferred(&mut self) -> &mut usize {
                &mut self.deferred
            }
            fn expr_call(&mut self, call: &syn::ExprCall) {
                if super::path_is(&call.func, &["QwenModel", "from_directory"]) {
                    self.calls += 1;
                }
            }
            fn macro_action(&mut self, _mac: &syn::Macro) -> MacroAction {
                MacroAction::Skip
            }
        }
        let mut work = Work::new(Probe::default());
        let expression: syn::Expr =
            syn::parse_str("vec![QwenModel::from_directory(path)]").expect("vec fixture");
        work.visit_expr(&expression);
        assert_eq!(work.calls, 1, "constructor within vec arguments");
        let repeat: syn::Expr = syn::parse_str("vec![QwenModel::from_directory(path); count()]")
            .expect("repeat fixture");
        work.visit_expr(&repeat);
        assert_eq!(work.calls, 2, "constructor within vec repetition");
        let matched: syn::Expr = syn::parse_str(
            "matches!(QwenModel::from_directory(path), Some(_) if QwenModel::from_directory(path))",
        )
        .expect("matches fixture");
        work.visit_expr(&matched);
        assert_eq!(work.calls, 4, "constructor in matches expression and guard");
        for (name, expected) in [("print", 5), ("std::print", 6), ("::std::print", 7)] {
            let printed: syn::Expr = syn::parse_str(&format!(
                "{name}!(\"{{:?}}\", QwenModel::from_directory(path))"
            ))
            .expect("print argument fixture");
            work.visit_expr(&printed);
            assert_eq!(work.calls, expected, "constructor in {name} argument");
        }
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
                MacroAction::Expressions(vec![syn::parse_str("target()").expect("target fixture")])
            }
        }
        let mut probe = Work::new(Probe::default());
        let invocation: syn::Expr = syn::parse_str("example!()").expect("macro fixture");
        probe.visit_expr(&invocation);
        assert_eq!(probe.calls, 1);
        let assignment: syn::Expr = syn::parse_str("left = target()").expect("assignment fixture");
        probe.visit_expr(&assignment);
        assert_eq!(probe.calls, 1);
    }
}
