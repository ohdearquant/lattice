// Shared source-analysis machinery for Cargo target discovery, cfg formulas,
// macro completeness, and Rust module source closures.
//
// Population closures follow literal `include!` paths and fail when a destination
// cannot be resolved. Callers that cannot analyze included macro content can still
// reject `include!` through `unclassifiable_include`.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::ext::IdentExt;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{Attribute, ItemFn, Meta, Token};

pub(crate) const CHECKED_CARGO_TARGET_KINDS: &[&str] = &["bench", "bin", "example", "lib", "test"];
pub(crate) fn rust_sources_under(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("read source directory") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(path);
            }
        }
    }
    sources.sort();
    sources
}

#[derive(Deserialize)]
pub(crate) struct CargoMetadata {
    pub(crate) packages: Vec<CargoPackage>,
}

#[derive(Deserialize)]
pub(crate) struct CargoPackage {
    pub(crate) manifest_path: PathBuf,
    pub(crate) targets: Vec<CargoMetadataTarget>,
}

#[derive(Deserialize)]
pub(crate) struct CargoMetadataTarget {
    pub(crate) name: String,
    pub(crate) kind: Vec<String>,
    pub(crate) src_path: PathBuf,
}

pub(crate) struct CargoTargetRoot {
    pub(crate) name: String,
    pub(crate) kind: String,
    pub(crate) path: PathBuf,
}

pub(crate) fn cargo_targets(
    manifest_dir: &Path,
    kinds: &[&str],
) -> Result<Vec<CargoTargetRoot>, String> {
    let manifest_path = manifest_dir.join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
            "--manifest-path",
        ])
        .arg(&manifest_path)
        .output()
        .map_err(|reason| format!("could not run cargo metadata: {reason}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed for {}: {}",
            manifest_path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let metadata: CargoMetadata = serde_json::from_slice(&output.stdout)
        .map_err(|reason| format!("cargo metadata output could not be parsed: {reason}"))?;
    let expected_manifest = std::fs::canonicalize(&manifest_path).map_err(|reason| {
        format!(
            "could not resolve package manifest {}: {reason}",
            manifest_path.display()
        )
    })?;
    let mut matching_packages = Vec::new();
    for package in metadata.packages {
        let package_manifest = std::fs::canonicalize(&package.manifest_path).map_err(|reason| {
            format!(
                "could not resolve cargo metadata manifest {}: {reason}",
                package.manifest_path.display()
            )
        })?;
        if package_manifest == expected_manifest {
            matching_packages.push(package);
        }
    }
    let mut packages = matching_packages.into_iter();
    let package = packages.next().ok_or_else(|| {
        format!(
            "cargo metadata did not contain package manifest {}",
            expected_manifest.display()
        )
    })?;
    if packages.next().is_some() {
        return Err(format!(
            "cargo metadata contained duplicate package manifest {}",
            expected_manifest.display()
        ));
    }

    let requested = kinds.iter().copied().collect::<BTreeSet<_>>();
    let checked_kinds = CHECKED_CARGO_TARGET_KINDS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut targets = Vec::new();
    for target in package.targets {
        if !target
            .kind
            .iter()
            .any(|kind| checked_kinds.contains(kind.as_str()))
        {
            return Err(format!(
                "cargo target {} has no checked target kind: {}",
                target.src_path.display(),
                target.kind.join(", ")
            ));
        }
        let matching = target
            .kind
            .iter()
            .filter(|kind| requested.contains(kind.as_str()))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            continue;
        }
        if matching.len() != 1 {
            return Err(format!(
                "cargo target {} has ambiguous requested kinds: {}",
                target.src_path.display(),
                matching
                    .iter()
                    .map(|kind| kind.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let path = std::fs::canonicalize(&target.src_path).map_err(|reason| {
            format!(
                "could not resolve Cargo target source {}: {reason}",
                target.src_path.display()
            )
        })?;
        targets.push(CargoTargetRoot {
            name: target.name,
            kind: matching[0].clone(),
            path,
        });
    }
    targets.sort_by(|left, right| left.path.cmp(&right.path));
    if targets.windows(2).any(|pair| pair[0].path == pair[1].path) {
        return Err("cargo metadata selected the same target source more than once".to_string());
    }
    Ok(targets)
}

pub(crate) fn cargo_target_roots(
    manifest_dir: &Path,
    kinds: &[&str],
) -> Result<Vec<PathBuf>, String> {
    cargo_targets(manifest_dir, kinds)
        .map(|targets| targets.into_iter().map(|target| target.path).collect())
}

#[derive(Clone, Debug)]
pub(crate) enum CfgFormula {
    True,
    False,
    Atom(String),
    Not(Box<CfgFormula>),
    All(Vec<CfgFormula>),
    Any(Vec<CfgFormula>),
}

impl CfgFormula {
    pub(crate) fn all(formulas: impl IntoIterator<Item = CfgFormula>) -> Self {
        let mut combined = Vec::new();
        for formula in formulas {
            match formula {
                Self::True => {}
                Self::False => return Self::False,
                Self::All(nested) => combined.extend(nested),
                other => combined.push(other),
            }
        }
        match combined.len() {
            0 => Self::True,
            1 => combined.pop().unwrap_or(Self::True),
            _ => Self::All(combined),
        }
    }

    pub(crate) fn any(formulas: impl IntoIterator<Item = CfgFormula>) -> Self {
        let mut combined = Vec::new();
        for formula in formulas {
            match formula {
                Self::False => {}
                Self::True => return Self::True,
                Self::Any(nested) => combined.extend(nested),
                other => combined.push(other),
            }
        }
        match combined.len() {
            0 => Self::False,
            1 => combined.pop().unwrap_or(Self::False),
            _ => Self::Any(combined),
        }
    }

    pub(crate) fn not(formula: CfgFormula) -> Self {
        match formula {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Not(inner) => *inner,
            other => Self::Not(Box::new(other)),
        }
    }

    pub(crate) fn collect_atoms(&self, atoms: &mut BTreeSet<String>) {
        match self {
            Self::Atom(atom) => {
                atoms.insert(atom.clone());
            }
            Self::Not(formula) => formula.collect_atoms(atoms),
            Self::All(formulas) | Self::Any(formulas) => {
                for formula in formulas {
                    formula.collect_atoms(atoms);
                }
            }
            Self::True | Self::False => {}
        }
    }

    pub(crate) fn evaluate(&self, values: &std::collections::BTreeMap<String, bool>) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::Atom(atom) => values.get(atom).copied().unwrap_or(false),
            Self::Not(formula) => !formula.evaluate(values),
            Self::All(formulas) => formulas.iter().all(|formula| formula.evaluate(values)),
            Self::Any(formulas) => formulas.iter().any(|formula| formula.evaluate(values)),
        }
    }

    pub(crate) fn satisfiable(&self) -> Result<bool, String> {
        self.satisfiable_with_test(None)
    }

    pub(crate) fn satisfiable_with_test(&self, test_cfg: Option<bool>) -> Result<bool, String> {
        let mut atoms = BTreeSet::new();
        self.collect_atoms(&mut atoms);
        if test_cfg.is_some() {
            atoms.remove("test");
        }
        if atoms.len() > 16 {
            return Err(format!(
                "cfg expression has {} independent atoms; refusing an unbounded classification",
                atoms.len()
            ));
        }
        let atoms = atoms.into_iter().collect::<Vec<_>>();
        for assignment in 0usize..(1usize << atoms.len()) {
            let mut values = atoms
                .iter()
                .enumerate()
                .map(|(index, atom)| (atom.clone(), assignment & (1 << index) != 0))
                .collect::<std::collections::BTreeMap<_, _>>();
            if let Some(value) = test_cfg {
                values.insert("test".to_string(), value);
            }
            if self.evaluate(&values) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn implies(&self, other: &CfgFormula) -> Result<bool, String> {
        Ok(!Self::all([self.clone(), Self::not(other.clone())]).satisfiable()?)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AttributeSpec {
    pub(crate) content: Range<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct FunctionSpec {
    pub(crate) name: String,
    pub(crate) body: Range<usize>,
    pub(crate) returns_unit: bool,
    pub(crate) attributes: Vec<AttributeSpec>,
    pub(crate) test_registration: TestRegistration,
    pub(crate) unclassifiable_macro: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) enum TestRegistration {
    No,
    Yes,
    Unclassifiable(String),
}

#[derive(Clone, Debug)]
pub(crate) struct ScopeSpec {
    pub(crate) body: Range<usize>,
    pub(crate) attributes: Vec<AttributeSpec>,
}

pub(crate) fn path_label(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

pub(crate) fn macro_tokens_name_protected_work(tokens: &proc_macro2::TokenStream) -> bool {
    let raw_dispatch = tokens.clone().into_iter().any(|token| match token {
        proc_macro2::TokenTree::Group(group) => macro_tokens_name_protected_work(&group.stream()),
        proc_macro2::TokenTree::Ident(ident) => matches!(
            ident.to_string().as_str(),
            "new_command_buffer"
                | "new_command_buffer_with_unretained_references"
                | "system_default"
        ),
        _ => false,
    });
    if raw_dispatch {
        return true;
    }
    let rendered = tokens.to_string();
    [
        "MetalErnie45State :: new",
        "MetalForwardPass :: new",
        "MetalQwen35State :: from_q4_dir",
        "MetalQwen35State :: new",
        "QwenModel :: from_directory",
    ]
    .iter()
    .any(|selector| rendered.contains(selector))
}

pub(crate) fn unclassifiable_macro(mac: &syn::Macro) -> Option<String> {
    let label = path_label(&mac.path);
    if label == "include" {
        return Some("unclassifiable include! in test-bearing scope".to_string());
    }
    let name = mac.path.segments.last()?.ident.to_string();
    if matches!(
        name.as_str(),
        "cfg"
            | "column"
            | "compile_error"
            | "concat"
            | "env"
            | "file"
            | "include_bytes"
            | "include_str"
            | "line"
            | "module_path"
            | "option_env"
            | "oslogstring"
            | "stringify"
    ) {
        // `compile_error!` expands to a diagnostic emitted at compile time:
        // it constructs nothing, calls nothing, and its one argument is a
        // string literal for the message -- there is no runtime dependency
        // it could carry, and no completeness gap in treating it as fully
        // classifiable (lattice PR-B addendum, 2026-09-20, lifting this
        // file's out-of-scope rule for this one classification arm).
        return None;
    }
    if matches!(
        name.as_str(),
        "assert"
            | "assert_eq"
            | "assert_ne"
            | "assert_relative_eq"
            | "dbg"
            | "debug_assert"
            | "debug_assert_eq"
            | "debug_assert_ne"
            | "debug"
            | "eprint"
            | "eprintln"
            | "format"
            | "format_args"
            | "is_aarch64_feature_detected"
            | "is_arm_feature_detected"
            | "is_x86_feature_detected"
            | "info"
            | "json"
            | "matches"
            | "panic"
            | "params"
            | "print"
            | "println"
            | "prop_assert"
            | "proptest"
            | "todo"
            | "thread_local"
            | "unimplemented"
            | "unreachable"
            | "value_parser"
            | "vec"
            | "warn"
            | "write"
            | "writeln"
    ) {
        return macro_tokens_name_protected_work(&mac.tokens).then(|| {
            format!("unclassifiable protected-work macro `{label}!` in test-bearing scope")
        });
    }
    Some(format!(
        "unclassifiable macro invocation `{label}!` in test-bearing scope"
    ))
}

pub(crate) struct FunctionMacroCollector {
    pub(crate) error: Option<String>,
}

impl<'ast> Visit<'ast> for FunctionMacroCollector {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if self.error.is_none() {
            self.error = unclassifiable_macro(mac);
        }
    }
}

pub(crate) struct ItemMacroCollector {
    pub(crate) error: Option<String>,
}

pub(crate) struct MacroCollector<'ast> {
    pub(crate) macros: Vec<&'ast syn::Macro>,
}

impl<'ast> Visit<'ast> for MacroCollector<'ast> {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.macros.push(mac);
    }
}

pub(crate) fn macro_delimiter(mac: &syn::Macro) -> (char, usize, char, usize) {
    match &mac.delimiter {
        syn::MacroDelimiter::Paren(token) => (
            '(',
            token.span.open().byte_range().start,
            ')',
            token.span.close().byte_range().start,
        ),
        syn::MacroDelimiter::Brace(token) => (
            '{',
            token.span.open().byte_range().start,
            '}',
            token.span.close().byte_range().start,
        ),
        syn::MacroDelimiter::Bracket(token) => (
            '[',
            token.span.open().byte_range().start,
            ']',
            token.span.close().byte_range().start,
        ),
    }
}

impl<'ast> Visit<'ast> for ItemMacroCollector {
    fn visit_item_fn(&mut self, _function: &'ast ItemFn) {}

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        if item.ident.is_none() && self.error.is_none() {
            self.error = unclassifiable_macro(&item.mac);
        }
    }
}

pub(crate) fn unclassifiable_item_macro(file: &syn::File) -> Option<String> {
    let mut collector = ItemMacroCollector { error: None };
    collector.visit_file(file);
    collector.error
}

pub(crate) fn unclassifiable_include(file: &syn::File, test_cfg: bool) -> Option<String> {
    let mut collector = MacroCollector { macros: Vec::new() };
    collector.visit_file(file);
    collector.macros.into_iter().find_map(|mac| {
        mac.path.is_ident("include").then(|| {
            if test_cfg {
                "unclassifiable include! in test-bearing scope".to_string()
            } else {
                "unclassifiable include! in Metal hazard scope".to_string()
            }
        })
    })
}

pub(crate) fn module_functions<'ast>(items: &'ast [syn::Item], functions: &mut Vec<&'ast ItemFn>) {
    for item in items {
        match item {
            syn::Item::Fn(function) => functions.push(function),
            syn::Item::Mod(module) => {
                if let Some((_, contents)) = &module.content {
                    module_functions(contents, functions);
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn inline_modules<'ast>(
    items: &'ast [syn::Item],
    modules: &mut Vec<&'ast syn::ItemMod>,
) {
    for item in items {
        if let syn::Item::Mod(module) = item
            && let Some((_, contents)) = &module.content
        {
            modules.push(module);
            inline_modules(contents, modules);
        }
    }
}

/// The receiver type name for an `impl` block, used as a path segment so
/// methods on different types with the same name (e.g. two `fn load`) do not
/// collide in a construction site's stable key.
pub(crate) fn impl_self_type_label(self_ty: &syn::Type) -> String {
    if let syn::Type::Path(type_path) = self_ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident.to_string();
    }
    "impl".to_string()
}

#[derive(Clone)]
pub(crate) struct ModuleSource {
    pub(crate) path: PathBuf,
    pub(crate) external_cfg: CfgFormula,
}

pub(crate) fn syn_meta_arguments(list: &syn::MetaList, context: &str) -> Result<Vec<Meta>, String> {
    Punctuated::<Meta, Token![,]>::parse_terminated
        .parse2(list.tokens.clone())
        .map(|arguments| arguments.into_iter().collect())
        .map_err(|reason| format!("{context}: unclassifiable cfg syntax: {reason}"))
}

pub(crate) fn syn_cfg_predicate(
    meta: &Meta,
    test_cfg: bool,
    context: &str,
) -> Result<CfgFormula, String> {
    syn_cfg_predicate_with_test(meta, Some(test_cfg), context)
}

fn syn_cfg_predicate_with_test(
    meta: &Meta,
    test_cfg: Option<bool>,
    context: &str,
) -> Result<CfgFormula, String> {
    match meta {
        Meta::Path(path) => {
            let label = path_label(path);
            Ok(match label.as_str() {
                "test" => match test_cfg {
                    Some(true) => CfgFormula::True,
                    Some(false) => CfgFormula::False,
                    None => CfgFormula::Atom(label),
                },
                "unix" if test_cfg.is_some() => CfgFormula::True,
                "windows" if test_cfg.is_some() => CfgFormula::False,
                _ => CfgFormula::Atom(label),
            })
        }
        Meta::NameValue(value) => {
            let label = path_label(&value.path);
            let syn::Expr::Lit(expression) = &value.value else {
                return Err(format!(
                    "{context}: cfg value for `{label}` is not a literal"
                ));
            };
            let syn::Lit::Str(value) = &expression.lit else {
                return Err(format!(
                    "{context}: cfg value for `{label}` is not a string"
                ));
            };
            let value = value.value();
            if test_cfg.is_none() {
                return Ok(CfgFormula::Atom(format!("{label}={value}")));
            }
            Ok(match (label.as_str(), value.as_str()) {
                ("target_os", "macos") | ("target_family", "unix") => CfgFormula::True,
                ("target_os", _) | ("target_family", "windows") => CfgFormula::False,
                ("feature", "metal-gpu") => CfgFormula::True,
                _ => CfgFormula::Atom(format!("{label}={value}")),
            })
        }
        Meta::List(list) => {
            let name = path_label(&list.path);
            let arguments = syn_meta_arguments(list, context)?;
            match name.as_str() {
                "all" => Ok(CfgFormula::all(
                    arguments
                        .iter()
                        .map(|argument| syn_cfg_predicate_with_test(argument, test_cfg, context))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "any" => Ok(CfgFormula::any(
                    arguments
                        .iter()
                        .map(|argument| syn_cfg_predicate_with_test(argument, test_cfg, context))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "not" if arguments.len() == 1 => Ok(CfgFormula::not(syn_cfg_predicate_with_test(
                    &arguments[0],
                    test_cfg,
                    context,
                )?)),
                "not" => Err(format!("{context}: cfg(not(...)) needs one predicate")),
                _ => Err(format!(
                    "{context}: unsupported cfg predicate `{name}(...)`"
                )),
            }
        }
    }
}

pub(crate) fn syn_cfg_effect(
    meta: &Meta,
    test_cfg: bool,
    context: &str,
) -> Result<CfgFormula, String> {
    syn_cfg_effect_with_test(meta, Some(test_cfg), context)
}

fn syn_cfg_effect_with_test(
    meta: &Meta,
    test_cfg: Option<bool>,
    context: &str,
) -> Result<CfgFormula, String> {
    let Meta::List(list) = meta else {
        let name = path_label(meta.path());
        return if matches!(name.as_str(), "cfg" | "cfg_attr") {
            Err(format!("{context}: malformed `{name}` attribute"))
        } else {
            Ok(CfgFormula::True)
        };
    };
    let name = path_label(&list.path);
    let arguments = syn_meta_arguments(list, context)?;
    if name == "cfg" {
        if arguments.len() != 1 {
            return Err(format!("{context}: cfg attribute needs one predicate"));
        }
        return match test_cfg {
            Some(test_cfg) => syn_cfg_predicate(&arguments[0], test_cfg, context),
            None => syn_cfg_predicate_with_test(&arguments[0], None, context),
        };
    }
    if name != "cfg_attr" {
        return Ok(CfgFormula::True);
    }
    if arguments.len() < 2 {
        return Err(format!(
            "{context}: cfg_attr attribute needs a predicate and emitted attribute"
        ));
    }
    let predicate = match test_cfg {
        Some(test_cfg) => syn_cfg_predicate(&arguments[0], test_cfg, context)?,
        None => syn_cfg_predicate_with_test(&arguments[0], None, context)?,
    };
    Ok(CfgFormula::all(
        arguments
            .iter()
            .skip(1)
            .map(|emitted| {
                Ok(CfgFormula::any([
                    CfgFormula::not(predicate.clone()),
                    syn_cfg_effect_with_test(emitted, test_cfg, context)?,
                ]))
            })
            .collect::<Result<Vec<_>, String>>()?,
    ))
}

pub(crate) fn syn_attributes_formula(
    attributes: &[Attribute],
    test_cfg: bool,
    context: &str,
) -> Result<CfgFormula, String> {
    Ok(CfgFormula::all(
        attributes
            .iter()
            .map(|attribute| syn_cfg_effect(&attribute.meta, test_cfg, context))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

pub(crate) fn syn_attributes_formula_with_test(
    attributes: &[Attribute],
    test_cfg: Option<bool>,
    context: &str,
) -> Result<CfgFormula, String> {
    Ok(CfgFormula::all(
        attributes
            .iter()
            .map(|attribute| syn_cfg_effect_with_test(&attribute.meta, test_cfg, context))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

pub(crate) fn cfg_attr_emits_path(meta: &Meta, context: &str) -> Result<bool, String> {
    let Meta::List(list) = meta else {
        return if meta.path().is_ident("cfg_attr") {
            Err(format!("{context}: malformed `cfg_attr` attribute"))
        } else {
            Ok(false)
        };
    };
    if !list.path.is_ident("cfg_attr") {
        return Ok(false);
    }
    for emitted in syn_meta_arguments(list, context)?.iter().skip(1) {
        if emitted.path().is_ident("path") || cfg_attr_emits_path(emitted, context)? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn literal_module_path(
    attributes: &[Attribute],
    context: &str,
) -> Result<Option<PathBuf>, String> {
    for attribute in attributes {
        if cfg_attr_emits_path(&attribute.meta, context)? {
            return Err(format!(
                "{context}: unclassifiable cfg_attr-generated module path"
            ));
        }
        if !attribute.path().is_ident("path") {
            continue;
        }
        let Meta::NameValue(value) = &attribute.meta else {
            return Err(format!("{context}: unclassifiable module path attribute"));
        };
        let syn::Expr::Lit(expression) = &value.value else {
            return Err(format!("{context}: module path is not a string literal"));
        };
        let syn::Lit::Str(path) = &expression.lit else {
            return Err(format!("{context}: module path is not a string literal"));
        };
        return Ok(Some(PathBuf::from(path.value())));
    }
    Ok(None)
}

pub(crate) fn item_attributes(item: &syn::Item) -> &[Attribute] {
    match item {
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Enum(item) => &item.attrs,
        syn::Item::ExternCrate(item) => &item.attrs,
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::ForeignMod(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        syn::Item::Macro(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Struct(item) => &item.attrs,
        syn::Item::Trait(item) => &item.attrs,
        syn::Item::TraitAlias(item) => &item.attrs,
        syn::Item::Type(item) => &item.attrs,
        syn::Item::Union(item) => &item.attrs,
        syn::Item::Use(item) => &item.attrs,
        syn::Item::Verbatim(_) => &[],
        _ => &[],
    }
}

struct IncludeTarget {
    path: PathBuf,
    cfg: CfgFormula,
}

struct IncludeCollector<'a> {
    test_cfg: Option<bool>,
    context: &'a str,
    inherited_cfg: CfgFormula,
    targets: Vec<IncludeTarget>,
    error: Option<String>,
}

impl IncludeCollector<'_> {
    fn with_attributes(&mut self, attributes: &[Attribute], visit: impl FnOnce(&mut Self)) {
        if self.error.is_some() {
            return;
        }
        let local_cfg =
            match syn_attributes_formula_with_test(attributes, self.test_cfg, self.context) {
                Ok(cfg) => cfg,
                Err(reason) => {
                    self.error = Some(reason);
                    return;
                }
            };
        let previous = self.inherited_cfg.clone();
        self.inherited_cfg = CfgFormula::all([previous.clone(), local_cfg]);
        match self.inherited_cfg.satisfiable() {
            Ok(true) => visit(self),
            Ok(false) => {}
            Err(reason) => self.error = Some(format!("{}: {reason}", self.context)),
        }
        self.inherited_cfg = previous;
    }
}

impl<'ast> Visit<'ast> for IncludeCollector<'_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        self.with_attributes(item_attributes(item), |collector| {
            syn::visit::visit_item(collector, item);
        });
    }

    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        let attributes: &[Attribute] = match item {
            syn::ImplItem::Const(item) => &item.attrs,
            syn::ImplItem::Fn(item) => &item.attrs,
            syn::ImplItem::Type(item) => &item.attrs,
            syn::ImplItem::Macro(item) => &item.attrs,
            syn::ImplItem::Verbatim(_) => &[],
            _ => &[],
        };
        self.with_attributes(attributes, |collector| {
            syn::visit::visit_impl_item(collector, item);
        });
    }

    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        let attributes: &[Attribute] = match item {
            syn::TraitItem::Const(item) => &item.attrs,
            syn::TraitItem::Fn(item) => &item.attrs,
            syn::TraitItem::Type(item) => &item.attrs,
            syn::TraitItem::Macro(item) => &item.attrs,
            syn::TraitItem::Verbatim(_) => &[],
            _ => &[],
        };
        self.with_attributes(attributes, |collector| {
            syn::visit::visit_trait_item(collector, item);
        });
    }

    fn visit_foreign_item(&mut self, item: &'ast syn::ForeignItem) {
        let attributes: &[Attribute] = match item {
            syn::ForeignItem::Fn(item) => &item.attrs,
            syn::ForeignItem::Static(item) => &item.attrs,
            syn::ForeignItem::Type(item) => &item.attrs,
            syn::ForeignItem::Macro(item) => &item.attrs,
            syn::ForeignItem::Verbatim(_) => &[],
            _ => &[],
        };
        self.with_attributes(attributes, |collector| {
            syn::visit::visit_foreign_item(collector, item);
        });
    }

    fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
        self.with_attributes(&node.attrs, |collector| {
            syn::visit::visit_expr_macro(collector, node);
        });
    }

    fn visit_stmt_macro(&mut self, node: &'ast syn::StmtMacro) {
        self.with_attributes(&node.attrs, |collector| {
            syn::visit::visit_stmt_macro(collector, node);
        });
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if !mac.path.is_ident("include") || self.error.is_some() {
            return;
        }
        match syn::parse2::<syn::LitStr>(mac.tokens.clone()) {
            Ok(path) => self.targets.push(IncludeTarget {
                path: PathBuf::from(path.value()),
                cfg: self.inherited_cfg.clone(),
            }),
            Err(reason) => {
                self.error = Some(format!(
                    "{}: include! path is not a string literal: {reason}",
                    self.context
                ));
            }
        }
    }
}

pub(crate) fn module_destination(
    module: &syn::ItemMod,
    module_dir: &Path,
    path_attribute_dir: &Path,
    context: &str,
) -> Result<(PathBuf, PathBuf), String> {
    let module_name = module.ident.unraw().to_string();
    let direct_path = literal_module_path(&module.attrs, context)?;
    let target = if let Some(relative) = &direct_path {
        path_attribute_dir.join(relative)
    } else {
        let flat = module_dir.join(format!("{module_name}.rs"));
        let nested = module_dir.join(&module_name).join("mod.rs");
        match (flat.exists(), nested.exists()) {
            (true, false) => flat,
            (false, true) => nested,
            (true, true) => {
                return Err(format!(
                    "{context}: ambiguous compiler-selected module `{}` at {} and {}",
                    module.ident,
                    flat.display(),
                    nested.display()
                ));
            }
            (false, false) => {
                return Err(format!(
                    "{context}: compiler-selected module `{}` could not be resolved at {} or {}",
                    module.ident,
                    flat.display(),
                    nested.display()
                ));
            }
        }
    };
    let child_module_dir =
        if direct_path.is_some() || target.file_name().is_some_and(|name| name == "mod.rs") {
            target.parent().unwrap_or(module_dir).to_path_buf()
        } else {
            target.parent().unwrap_or(module_dir).join(&module_name)
        };
    Ok((target, child_module_dir))
}

pub(crate) struct ModuleGraph<'a> {
    pub(crate) manifest_dir: &'a Path,
    pub(crate) manifest_canonical: PathBuf,
    pub(crate) test_cfg: Option<bool>,
    pub(crate) follow_includes: bool,
    pub(crate) sources: BTreeMap<PathBuf, CfgFormula>,
    pub(crate) visiting: BTreeSet<PathBuf>,
}

impl ModuleGraph<'_> {
    pub(crate) fn load(
        &mut self,
        path: &Path,
        module_dir: &Path,
        inherited_cfg: CfgFormula,
    ) -> Result<(), String> {
        let canonical = std::fs::canonicalize(path).map_err(|reason| {
            format!(
                "unclassifiable compiler-selected module {}: {reason}",
                path.display()
            )
        })?;
        if !canonical.starts_with(&self.manifest_canonical) {
            return Err(format!(
                "unclassifiable compiler-selected module outside crate boundary: {}",
                canonical.display()
            ));
        }
        if !self.visiting.insert(canonical.clone()) {
            return Err(format!(
                "module graph cycle while classifying {}",
                canonical.display()
            ));
        }
        let source = std::fs::read_to_string(&canonical).map_err(|reason| {
            format!(
                "could not read compiler-selected module {}: {reason}",
                canonical.display()
            )
        })?;
        let context = canonical
            .strip_prefix(self.manifest_dir)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();
        let syntax = syn::parse_file(&source).map_err(|reason| {
            format!("{context}: Rust syntax could not be classified: {reason}")
        })?;
        let file_cfg = CfgFormula::all([
            inherited_cfg,
            syn_attributes_formula_with_test(&syntax.attrs, self.test_cfg, &context)?,
        ]);
        if !file_cfg.satisfiable()? {
            self.visiting.remove(&canonical);
            return Ok(());
        }
        if let Some(existing) = self.sources.get_mut(&canonical) {
            if file_cfg.implies(existing)? {
                self.visiting.remove(&canonical);
                return Ok(());
            }
            *existing = CfgFormula::any([existing.clone(), file_cfg.clone()]);
        } else {
            self.sources.insert(canonical.clone(), file_cfg.clone());
        }
        self.walk_items(
            &syntax.items,
            &canonical,
            module_dir,
            path.parent().unwrap_or(module_dir),
            file_cfg,
        )?;
        self.visiting.remove(&canonical);
        Ok(())
    }

    pub(crate) fn walk_items(
        &mut self,
        items: &[syn::Item],
        source_path: &Path,
        module_dir: &Path,
        path_attribute_dir: &Path,
        inherited_cfg: CfgFormula,
    ) -> Result<(), String> {
        let context = source_path
            .strip_prefix(self.manifest_dir)
            .unwrap_or(source_path)
            .to_string_lossy()
            .into_owned();
        for item in items {
            let syn::Item::Mod(module) = item else {
                if self.follow_includes {
                    let mut collector = IncludeCollector {
                        test_cfg: self.test_cfg,
                        context: &context,
                        inherited_cfg: inherited_cfg.clone(),
                        targets: Vec::new(),
                        error: None,
                    };
                    collector.visit_item(item);
                    if let Some(reason) = collector.error {
                        return Err(reason);
                    }
                    for include in collector.targets {
                        let containing_dir = source_path.parent().unwrap_or(self.manifest_dir);
                        let target = containing_dir.join(include.path);
                        self.load(&target, module_dir, include.cfg)?;
                    }
                }
                continue;
            };
            // A raw identifier (`mod r#loop;`) names the file `loop.rs`, not `r#loop.rs`.
            let module_name = module.ident.unraw().to_string();
            let module_cfg = CfgFormula::all([
                inherited_cfg.clone(),
                syn_attributes_formula_with_test(&module.attrs, self.test_cfg, &context)?,
            ]);
            if !module_cfg.satisfiable()? {
                continue;
            }
            if let Some((_, contents)) = &module.content {
                if literal_module_path(&module.attrs, &context)?.is_some() {
                    return Err(format!(
                        "{context}: unclassifiable path attribute on inline module `{}`",
                        module.ident
                    ));
                }
                let inline_dir = module_dir.join(&module_name);
                self.walk_items(contents, source_path, &inline_dir, &inline_dir, module_cfg)?;
                continue;
            }
            let (target, child_module_dir) =
                module_destination(module, module_dir, path_attribute_dir, &context)?;
            self.load(&target, &child_module_dir, module_cfg)?;
        }
        Ok(())
    }
}

/// Returns the compiler-selected source closure for one Cargo target root.
///
/// The serialization boundary is bounded repository-owned tests, benches,
/// examples, and measurement binaries that can submit Metal work, not a
/// physical `src/**/*.rs` walk. Production modules are covered through those
/// callers because they cannot execute independently. Long-running interactive
/// and service targets are explicit exemptions: they must acquire the same
/// fleet lock externally when used as measurements. The crate has no build
/// script or executable Metal doctest, so neither is a current hazard entrypoint.
pub(crate) fn module_source_closure(
    manifest_dir: &Path,
    root: &Path,
    test_cfg: bool,
) -> Result<Vec<ModuleSource>, String> {
    let module_dir = root
        .parent()
        .ok_or_else(|| format!("target root {} has no parent", root.display()))?;
    module_source_closure_with_mode(manifest_dir, root, module_dir, Some(test_cfg), false)
}

/// Returns a source closure with symbolic configuration predicates and literal includes enabled.
/// The supplied module directory is the directory Rust uses for external modules declared by
/// `root`, which differs from `root.parent()` when the root is itself a module file.
#[allow(
    dead_code,
    reason = "Shared support is compiled by separate integration test targets"
)]
pub(crate) fn module_source_population_closure(
    manifest_dir: &Path,
    root: &Path,
    root_module_dir: &Path,
) -> Result<Vec<ModuleSource>, String> {
    module_source_closure_with_mode(manifest_dir, root, root_module_dir, None, true)
}

fn module_source_closure_with_mode(
    manifest_dir: &Path,
    root: &Path,
    root_module_dir: &Path,
    test_cfg: Option<bool>,
    follow_includes: bool,
) -> Result<Vec<ModuleSource>, String> {
    let manifest_canonical = std::fs::canonicalize(manifest_dir).map_err(|reason| {
        format!(
            "could not resolve crate boundary {}: {reason}",
            manifest_dir.display()
        )
    })?;
    let mut graph = ModuleGraph {
        manifest_dir,
        manifest_canonical,
        test_cfg,
        follow_includes,
        sources: BTreeMap::new(),
        visiting: BTreeSet::new(),
    };
    graph.load(root, root_module_dir, CfgFormula::True)?;
    Ok(graph
        .sources
        .into_iter()
        .map(|(path, external_cfg)| ModuleSource { path, external_cfg })
        .collect())
}
