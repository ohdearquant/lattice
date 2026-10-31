//! Source contract for the Metal-capable model initialization in `layer_sweep`.

use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{Attribute, Expr, Item, ItemUse, Local, Meta, Pat, Stmt, Token, UseTree};

const SOURCE: &str = include_str!("../examples/layer_sweep.rs");
const FACADE_IMPORT: &str = "use lattice_inference::QwenModel;";
const OWNER_IMPORT: &str = "use lattice_model_qwen3_embedding::QwenModel;";
const GUARD: &str = "let _gpu_lock = lattice_inference::measurement::gpu_test_lock();";
const CONSTRUCT: &str = "let model = QwenModel::from_directory(dir).unwrap();";

fn path_is(path: &syn::Path, expected: &[&str]) -> bool {
    path.leading_colon.is_none()
        && path.segments.len() == expected.len()
        && path
            .segments
            .iter()
            .zip(expected)
            .all(|(segment, name)| segment.ident == *name && segment.arguments.is_empty())
}

fn metal_cfg(attrs: &[Attribute]) -> bool {
    let [attr] = attrs else { return false };
    let Meta::List(cfg) = &attr.meta else {
        return false;
    };
    if !cfg.path.is_ident("cfg") {
        return false;
    }
    let Ok(Meta::List(all)) = syn::parse2::<Meta>(cfg.tokens.clone()) else {
        return false;
    };
    if !all.path.is_ident("all") {
        return false;
    }
    let Ok(parts) = all.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) else {
        return false;
    };
    let mut observed = BTreeSet::new();
    for part in parts {
        let Meta::NameValue(pair) = part else {
            return false;
        };
        let Some(name) = pair.path.get_ident() else {
            return false;
        };
        let Expr::Lit(value) = pair.value else {
            return false;
        };
        let syn::Lit::Str(value) = value.lit else {
            return false;
        };
        if !observed.insert((name.to_string(), value.value())) {
            return false;
        }
    }
    observed
        == BTreeSet::from([
            ("feature".to_string(), "metal-gpu".to_string()),
            ("target_os".to_string(), "macos".to_string()),
        ])
}

fn ident_pattern(local: &Local, expected: &str) -> bool {
    matches!(&local.pat, Pat::Ident(pat) if pat.ident == expected
        && pat.by_ref.is_none() && pat.mutability.is_none() && pat.subpat.is_none())
}

fn guarded_lock(local: &Local) -> bool {
    if !ident_pattern(local, "_gpu_lock") || !metal_cfg(&local.attrs) {
        return false;
    }
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::Call(call) = init.expr.as_ref() else {
        return false;
    };
    let Expr::Path(function) = call.func.as_ref() else {
        return false;
    };
    init.diverge.is_none()
        && call.args.is_empty()
        && path_is(
            &function.path,
            &["lattice_inference", "measurement", "gpu_test_lock"],
        )
}

fn model_construction(local: &Local) -> bool {
    if !ident_pattern(local, "model") || !local.attrs.is_empty() {
        return false;
    }
    let Some(init) = &local.init else {
        return false;
    };
    let Expr::MethodCall(unwrap) = init.expr.as_ref() else {
        return false;
    };
    if init.diverge.is_some() || unwrap.method != "unwrap" || !unwrap.args.is_empty() {
        return false;
    }
    let Expr::Call(call) = unwrap.receiver.as_ref() else {
        return false;
    };
    let Expr::Path(function) = call.func.as_ref() else {
        return false;
    };
    let Some(Expr::Path(dir)) = call.args.first() else {
        return false;
    };
    call.args.len() == 1
        && path_is(&function.path, &["QwenModel", "from_directory"])
        && path_is(&dir.path, &["dir"])
}

fn contains_sensitive_tokens(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Group(group) => contains_sensitive_tokens(group.stream()),
        TokenTree::Ident(ident) => {
            ident == "_gpu_lock" || ident == "QwenModel" || ident == "from_directory"
        }
        TokenTree::Literal(literal) => {
            let value = literal.to_string();
            value.contains("_gpu_lock")
                || value.contains("QwenModel")
                || value.contains("from_directory")
        }
        TokenTree::Punct(_) => false,
    })
}

fn direct_qwen_import(item: &ItemUse) -> bool {
    if !item.attrs.is_empty() || item.leading_colon.is_some() {
        return false;
    }
    let UseTree::Path(parent) = &item.tree else {
        return false;
    };
    if parent.ident != "lattice_inference" && parent.ident != "lattice_model_qwen3_embedding" {
        return false;
    }
    matches!(parent.tree.as_ref(), UseTree::Name(name) if name.ident == "QwenModel")
}

#[derive(Default)]
struct Audit {
    imports: usize,
    invalid_import: bool,
    lock_bindings: usize,
    guard_uses: usize,
    constructor_references: usize,
    invalid_constructor: bool,
    lock_calls: usize,
    invalid_lock_call: bool,
    shadowed_model_type: bool,
    sensitive_macro: bool,
}

impl<'ast> Visit<'ast> for Audit {
    fn visit_item_use(&mut self, item: &'ast ItemUse) {
        fn contains_qwen(tree: &UseTree) -> bool {
            match tree {
                UseTree::Path(path) => path.ident == "QwenModel" || contains_qwen(&path.tree),
                UseTree::Name(name) => name.ident == "QwenModel",
                UseTree::Rename(rename) => {
                    rename.ident == "QwenModel" || rename.rename == "QwenModel"
                }
                UseTree::Group(group) => group.items.iter().any(contains_qwen),
                UseTree::Glob(_) => false,
            }
        }
        if contains_qwen(&item.tree) {
            self.imports += 1;
            self.invalid_import |= !direct_qwen_import(item);
        }
        syn::visit::visit_item_use(self, item);
    }

    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.lock_bindings += usize::from(pat.ident == "_gpu_lock");
        self.shadowed_model_type |= pat.ident == "QwenModel";
        syn::visit::visit_pat_ident(self, pat);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.guard_uses += usize::from(path.path.segments.iter().any(|s| s.ident == "_gpu_lock"));
        if path
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "from_directory")
        {
            self.constructor_references += 1;
            self.invalid_constructor |= !path_is(&path.path, &["QwenModel", "from_directory"]);
        }
        syn::visit::visit_expr_path(self, path);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let Expr::Path(path) = call.func.as_ref()
            && let Some(last) = path.path.segments.last()
            && last.ident == "gpu_test_lock"
        {
            self.lock_calls += 1;
            self.invalid_lock_call |= !path_is(
                &path.path,
                &["lattice_inference", "measurement", "gpu_test_lock"],
            );
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "from_directory" || call.method == "gpu_test_lock" {
            self.invalid_constructor = true;
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.shadowed_model_type |= item.ident == "QwenModel";
        syn::visit::visit_item_struct(self, item);
    }

    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        self.shadowed_model_type |= item.ident == "QwenModel";
        syn::visit::visit_item_enum(self, item);
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.shadowed_model_type |= item.ident == "QwenModel";
        syn::visit::visit_item_type(self, item);
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        self.shadowed_model_type |= item.ident == "QwenModel";
        syn::visit::visit_item_mod(self, item);
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        self.shadowed_model_type |= item.ident == "QwenModel";
        syn::visit::visit_item_trait(self, item);
    }

    fn visit_item_union(&mut self, item: &'ast syn::ItemUnion) {
        self.shadowed_model_type |= item.ident == "QwenModel";
        syn::visit::visit_item_union(self, item);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.sensitive_macro |= contains_sensitive_tokens(mac.tokens.clone());
    }
}

fn validate(source: &str) -> Result<(), &'static str> {
    let file = syn::parse_file(source).map_err(|_| "source does not parse")?;
    let mains: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "main" => Some(function),
            _ => None,
        })
        .collect();
    let [main] = mains.as_slice() else {
        return Err("main is missing or duplicated");
    };
    if !main.attrs.is_empty() {
        return Err("main has unexpected attributes");
    }
    let guards: Vec<_> = main
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(index, stmt)| match stmt {
            Stmt::Local(local) if guarded_lock(local) => Some(index),
            _ => None,
        })
        .collect();
    let constructors: Vec<_> = main
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(index, stmt)| match stmt {
            Stmt::Local(local) if model_construction(local) => Some(index),
            _ => None,
        })
        .collect();
    let ([guard], [constructor]) = (guards.as_slice(), constructors.as_slice()) else {
        return Err("guard or model constructor changed");
    };
    if guard >= constructor {
        return Err("model construction precedes guard");
    }
    let mut audit = Audit::default();
    audit.visit_file(&file);
    if audit.imports != 1 || audit.invalid_import || audit.shadowed_model_type {
        return Err("QwenModel binding changed");
    }
    if audit.lock_bindings != 1 || audit.guard_uses != 0 || audit.sensitive_macro {
        return Err("guard or constructor escaped or was shadowed");
    }
    if audit.constructor_references != 1 || audit.invalid_constructor {
        return Err("model constructor changed or duplicated");
    }
    if audit.lock_calls != 1 || audit.invalid_lock_call {
        return Err("lock acquisition changed or duplicated");
    }
    Ok(())
}

fn replace_once(source: &str, before: &str, after: &str) -> String {
    assert_eq!(
        source.matches(before).count(),
        1,
        "mutation anchor {before}"
    );
    source.replacen(before, after, 1)
}

fn current_import() -> (&'static str, &'static str) {
    match (
        SOURCE.matches(FACADE_IMPORT).count(),
        SOURCE.matches(OWNER_IMPORT).count(),
    ) {
        (1, 0) => (FACADE_IMPORT, OWNER_IMPORT),
        (0, 1) => (OWNER_IMPORT, FACADE_IMPORT),
        _ => panic!("unexpected QwenModel import"),
    }
}

#[test]
fn layer_sweep_metal_capable_construction_keeps_the_guard() {
    assert_eq!(validate(SOURCE), Ok(()));
    let (current, alternative) = current_import();
    let alternative_source = replace_once(SOURCE, current, alternative);
    assert_eq!(validate(&alternative_source), Ok(()));
}

#[test]
fn layer_sweep_contract_rejects_guard_and_constructor_mutations() {
    let scoped = format!("{{ {GUARD} }}");
    let misplaced = format!(
        "{CONSTRUCT}\n    #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\n    {GUARD}"
    );
    let full = format!(
        "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\n    {GUARD}\n    {CONSTRUCT}"
    );
    let mutants = [
        (
            "removed lock",
            replace_once(SOURCE, GUARD, "let _gpu_lock = ();"),
            "guard or model constructor changed",
        ),
        (
            "narrowed scope",
            replace_once(SOURCE, GUARD, &scoped),
            "guard or model constructor changed",
        ),
        (
            "late lock",
            replace_once(SOURCE, &full, &misplaced),
            "model construction precedes guard",
        ),
        (
            "explicit drop",
            replace_once(
                SOURCE,
                CONSTRUCT,
                &format!("{CONSTRUCT}\n    drop(_gpu_lock);"),
            ),
            "guard or constructor escaped or was shadowed",
        ),
        (
            "lock shadow",
            replace_once(
                SOURCE,
                CONSTRUCT,
                &format!("{CONSTRUCT}\n    let _gpu_lock = ();"),
            ),
            "guard or constructor escaped or was shadowed",
        ),
        (
            "macro capture",
            replace_once(
                SOURCE,
                CONSTRUCT,
                &format!("{CONSTRUCT}\n    println!(\"{{_gpu_lock:?}}\");"),
            ),
            "guard or constructor escaped or was shadowed",
        ),
        (
            "deferred second constructor",
            replace_once(
                SOURCE,
                CONSTRUCT,
                &format!("{CONSTRUCT}\n    let _deferred = || QwenModel::from_directory(dir);"),
            ),
            "model constructor changed or duplicated",
        ),
        (
            "constructor function binding",
            replace_once(
                SOURCE,
                CONSTRUCT,
                &format!("{CONSTRUCT}\n    let b = QwenModel::from_directory; let _ = b(dir);"),
            ),
            "model constructor changed or duplicated",
        ),
        (
            "constructor function passed to map",
            replace_once(
                SOURCE,
                CONSTRUCT,
                &format!("{CONSTRUCT}\n    let _ = Some(dir).map(QwenModel::from_directory);"),
            ),
            "model constructor changed or duplicated",
        ),
        (
            "different constructor",
            replace_once(
                SOURCE,
                CONSTRUCT,
                "let model = OtherModel::from_directory(dir).unwrap();",
            ),
            "guard or model constructor changed",
        ),
        (
            "renamed import",
            replace_once(
                SOURCE,
                current_import().0,
                "use lattice_inference::QwenModel as OtherModel;",
            ),
            "QwenModel binding changed",
        ),
        (
            "shadowed type",
            replace_once(
                SOURCE,
                CONSTRUCT,
                &format!("struct QwenModel;\n    {CONSTRUCT}"),
            ),
            "QwenModel binding changed",
        ),
        (
            "weakened cfg",
            replace_once(
                SOURCE,
                "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
                "#[cfg(target_os = \"macos\")]",
            ),
            "guard or model constructor changed",
        ),
    ];
    for (name, mutant, expected) in mutants {
        assert!(
            syn::parse_file(&mutant).is_ok(),
            "mutation must parse: {name}"
        );
        assert_eq!(validate(&mutant), Err(expected), "mutation: {name}");
    }
}
