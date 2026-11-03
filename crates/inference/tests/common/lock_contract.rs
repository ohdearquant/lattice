use syn::{Attribute, Expr, Meta, Pat, Stmt};

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
    use super::{guard, metal_cfg};
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
}
