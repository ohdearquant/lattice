use syn::visit::{self, Visit};
use syn::{Attribute, Expr, Meta, Pat, Stmt};

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

#[derive(Default)]
struct MacroAudit {
    bad: bool,
}

impl<'ast> Visit<'ast> for MacroAudit {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let is_test_module = item.attrs.iter().any(|attr| {
            attr.path().is_ident("cfg")
                && matches!(attr.parse_args::<Meta>(), Ok(Meta::Path(path)) if path.is_ident("test"))
        });
        if !is_test_module {
            visit::visit_item_mod(self, item);
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.bad |= shadowing_import(&item.tree);
        visit::visit_item_use(self, item);
    }

    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        self.bad |= attr.path().is_ident("macro_use");
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
    use super::{guard, macro_policy, metal_cfg, sensitive_tokens};
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
}
