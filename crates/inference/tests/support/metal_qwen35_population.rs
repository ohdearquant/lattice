//! Population coverage for the Qwen 3.5 Metal source tree.

use crate::source_graph::{
    CfgFormula, module_source_population_closure, syn_attributes_formula_with_test,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::Attribute;
use syn::visit::Visit;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ProductionItem {
    pub(crate) file: String,
    pub(crate) qualified_name: String,
    pub(crate) kind: String,
    pub(crate) ordinal: usize,
    pub(crate) reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FileKind {
    Production,
    TestOnly,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FileClassification {
    pub(crate) file: String,
    pub(crate) kind: FileKind,
    pub(crate) reason: String,
}

#[derive(Clone, Debug)]
pub(crate) struct PopulationManifest {
    pub(crate) discovered_files: Vec<String>,
    pub(crate) classifications: Vec<FileClassification>,
    pub(crate) production_items: Vec<ProductionItem>,
    pub(crate) reconciliation_failures: Vec<String>,
}

impl PopulationManifest {
    pub(crate) fn require_production_items_read(
        &self,
        read_items: &[ProductionItem],
    ) -> Result<(), String> {
        let read = read_items.iter().cloned().collect::<BTreeSet<_>>();
        let missing = self
            .production_items
            .iter()
            .filter(|item| !read.contains(*item))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        let files = missing
            .iter()
            .map(|item| item.file.as_str())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        Err(format!(
            "production item coverage is incomplete; unread items remain in {}",
            files.join(", ")
        ))
    }
}

pub(crate) fn derive_population(
    crate_dir: &Path,
    root: &Path,
    subtree: &Path,
) -> Result<PopulationManifest, String> {
    let crate_dir = crate_dir.canonicalize().map_err(|reason| {
        format!(
            "could not resolve crate directory {}: {reason}",
            crate_dir.display()
        )
    })?;
    let root = resolve_from(&crate_dir, root)
        .canonicalize()
        .map_err(|reason| {
            format!(
                "could not resolve population root {}: {reason}",
                root.display()
            )
        })?;
    let subtree = resolve_from(&crate_dir, subtree)
        .canonicalize()
        .map_err(|reason| {
            format!(
                "could not resolve population subtree {}: {reason}",
                subtree.display()
            )
        })?;
    if !root.starts_with(&crate_dir) {
        return Err(format!(
            "population root is outside the crate: {}",
            root.display()
        ));
    }
    if !subtree.starts_with(&crate_dir) {
        return Err(format!(
            "population subtree is outside the crate: {}",
            subtree.display()
        ));
    }

    let root_module_dir = root_module_dir(&root)?;
    let sources = module_source_population_closure(&crate_dir, &root, &root_module_dir)?;
    let source_cfg = sources
        .into_iter()
        .map(|source| (source.path, source.external_cfg))
        .collect::<BTreeMap<_, _>>();
    let graph_files = source_cfg.keys().cloned().collect::<BTreeSet<_>>();
    let physical_files = physical_rust_files(&subtree)?;
    let unreachable = physical_files
        .difference(&graph_files)
        .map(|path| relative_path(&crate_dir, path))
        .collect::<Vec<_>>();
    if !unreachable.is_empty() {
        return Err(format!(
            "Rust files under the population subtree are unreachable from its module graph: {}",
            unreachable.join(", ")
        ));
    }

    let mut discovered_files = graph_files
        .iter()
        .map(|path| relative_path(&crate_dir, path))
        .collect::<Vec<_>>();
    discovered_files.sort();

    let mut classifications = Vec::new();
    let mut production_items = Vec::new();
    for path in &graph_files {
        let relative = relative_path(&crate_dir, path);
        let source = fs::read_to_string(path)
            .map_err(|reason| format!("could not read discovered source {relative}: {reason}"))?;
        let syntax = syn::parse_file(&source)
            .map_err(|reason| format!("could not parse discovered source {relative}: {reason}"))?;
        let external_cfg = source_cfg
            .get(path)
            .ok_or_else(|| format!("discovered source has no module configuration: {relative}"))?;
        let mut reader = ItemReader {
            file: relative.clone(),
            inherited_cfg: external_cfg.clone(),
            scope: Vec::new(),
            next_ordinal: 0,
            production_items: Vec::new(),
            test_only_items: Vec::new(),
            error: None,
        };
        reader.visit_file(&syntax);
        if let Some(reason) = reader.error {
            return Err(reason);
        }
        let production_count = reader.production_items.len();
        let test_only_count = reader.test_only_items.len();
        if production_count > 0 {
            classifications.push(FileClassification {
                file: relative,
                kind: FileKind::Production,
                reason: format!("contains {production_count} item(s) reachable when test is false"),
            });
            production_items.extend(reader.production_items);
        } else if test_only_count > 0 {
            let example = reader
                .test_only_items
                .first()
                .map(|item| item.qualified_name.as_str())
                .unwrap_or("unknown item");
            classifications.push(FileClassification {
                file: relative,
                kind: FileKind::TestOnly,
                reason: format!(
                    "all discovered items require test configuration; example: {example}"
                ),
            });
        } else {
            let test_only_source = external_cfg
                .satisfiable_with_test(Some(false))
                .map(|reachable| !reachable)
                .map_err(|reason| format!("{relative}: {reason}"))?;
            classifications.push(FileClassification {
                file: relative,
                kind: if test_only_source {
                    FileKind::TestOnly
                } else {
                    FileKind::Production
                },
                reason: if test_only_source {
                    "source requires test configuration and contains no reachable items".into()
                } else {
                    "no items were found; retained because test-only status was not proven".into()
                },
            });
        }
    }
    classifications.sort_by(|left, right| left.file.cmp(&right.file));
    production_items.sort();

    Ok(PopulationManifest {
        discovered_files,
        classifications,
        production_items,
        reconciliation_failures: Vec::new(),
    })
}

fn resolve_from(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn root_module_dir(root: &Path) -> Result<PathBuf, String> {
    let parent = root
        .parent()
        .ok_or_else(|| format!("population root has no parent: {}", root.display()))?;
    if root.file_name().is_some_and(|name| name == "mod.rs") {
        return Ok(parent.to_path_buf());
    }
    let stem = root
        .file_stem()
        .ok_or_else(|| format!("population root has no file stem: {}", root.display()))?;
    Ok(parent.join(stem))
}

fn relative_path(crate_dir: &Path, path: &Path) -> String {
    path.strip_prefix(crate_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn physical_rust_files(root: &Path) -> Result<BTreeSet<PathBuf>, String> {
    let mut pending = vec![root.to_path_buf()];
    let mut visited_dirs = BTreeSet::new();
    let mut files = BTreeSet::new();
    while let Some(dir) = pending.pop() {
        let canonical_dir = dir.canonicalize().map_err(|reason| {
            format!(
                "could not resolve subtree directory {}: {reason}",
                dir.display()
            )
        })?;
        if !visited_dirs.insert(canonical_dir.clone()) {
            continue;
        }
        let entries = fs::read_dir(&canonical_dir).map_err(|reason| {
            format!(
                "could not read subtree directory {}: {reason}",
                canonical_dir.display()
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|reason| {
                format!(
                    "could not read an entry under {}: {reason}",
                    canonical_dir.display()
                )
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|reason| {
                format!(
                    "could not inspect subtree entry {}: {reason}",
                    path.display()
                )
            })?;
            if file_type.is_symlink() {
                return Err(format!(
                    "symbolic link under the population subtree is ambiguous: {}",
                    path.display()
                ));
            }
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
                let canonical = path.canonicalize().map_err(|reason| {
                    format!(
                        "could not resolve subtree source {}: {reason}",
                        path.display()
                    )
                })?;
                files.insert(canonical);
            }
        }
    }
    Ok(files)
}

struct ItemReader {
    file: String,
    inherited_cfg: CfgFormula,
    scope: Vec<String>,
    next_ordinal: usize,
    production_items: Vec<ProductionItem>,
    test_only_items: Vec<ProductionItem>,
    error: Option<String>,
}

impl ItemReader {
    fn visit_scoped(
        &mut self,
        attributes: &[Attribute],
        kind: &str,
        name: String,
        test_marker: bool,
        visit: impl FnOnce(&mut Self),
    ) {
        if self.error.is_some() {
            return;
        }
        let local_cfg = match syn_attributes_formula_with_test(attributes, None, &self.file) {
            Ok(cfg) => cfg,
            Err(reason) => {
                self.error = Some(reason);
                return;
            }
        };
        let effective_cfg = CfgFormula::all([self.inherited_cfg.clone(), local_cfg]);
        let reachable_without_test = match effective_cfg.satisfiable_with_test(Some(false)) {
            Ok(value) => value,
            Err(reason) => {
                self.error = Some(format!("{}: {reason}", self.file));
                return;
            }
        };
        let reachable_with_test = match effective_cfg.satisfiable_with_test(Some(true)) {
            Ok(value) => value,
            Err(reason) => {
                self.error = Some(format!("{}: {reason}", self.file));
                return;
            }
        };
        if !reachable_without_test && !reachable_with_test {
            return;
        }

        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        let mut path = self.scope.clone();
        path.push(name);
        let item = ProductionItem {
            file: self.file.clone(),
            qualified_name: path.join("::"),
            kind: kind.to_string(),
            ordinal,
            reason: "cfg is satisfiable with test false".to_string(),
        };
        if test_marker || (!reachable_without_test && reachable_with_test) {
            self.test_only_items.push(item);
        } else if reachable_without_test {
            self.production_items.push(item);
        }
        let previous_cfg = self.inherited_cfg.clone();
        self.inherited_cfg = if test_marker {
            CfgFormula::False
        } else {
            effective_cfg
        };
        self.scope = path;
        visit(self);
        self.scope.pop();
        self.inherited_cfg = previous_cfg;
    }
}

impl<'ast> Visit<'ast> for ItemReader {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if matches!(item, syn::Item::Verbatim(_)) {
            self.error = Some(format!(
                "{}: Rust item syntax could not be classified",
                self.file
            ));
            return;
        }
        let (kind, name, test_marker) = item_identity(item, self.next_ordinal);
        self.visit_scoped(
            crate::source_graph::item_attributes(item),
            kind,
            name,
            test_marker,
            |reader| syn::visit::visit_item(reader, item),
        );
    }

    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        if matches!(item, syn::ImplItem::Verbatim(_)) {
            self.error = Some(format!(
                "{}: implementation item syntax could not be classified",
                self.file
            ));
            return;
        }
        let (attributes, kind, name, test_marker) = impl_item_identity(item, self.next_ordinal);
        self.visit_scoped(attributes, kind, name, test_marker, |reader| {
            syn::visit::visit_impl_item(reader, item);
        });
    }

    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        if matches!(item, syn::TraitItem::Verbatim(_)) {
            self.error = Some(format!(
                "{}: trait item syntax could not be classified",
                self.file
            ));
            return;
        }
        let (attributes, kind, name, test_marker) = trait_item_identity(item, self.next_ordinal);
        self.visit_scoped(attributes, kind, name, test_marker, |reader| {
            syn::visit::visit_trait_item(reader, item);
        });
    }

    fn visit_foreign_item(&mut self, item: &'ast syn::ForeignItem) {
        if matches!(item, syn::ForeignItem::Verbatim(_)) {
            self.error = Some(format!(
                "{}: foreign item syntax could not be classified",
                self.file
            ));
            return;
        }
        let (attributes, kind, name, test_marker) = foreign_item_identity(item, self.next_ordinal);
        self.visit_scoped(attributes, kind, name, test_marker, |reader| {
            syn::visit::visit_foreign_item(reader, item);
        });
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if self.error.is_some() || mac.path.is_ident("include") {
            return;
        }
        let reachable_without_test = match self.inherited_cfg.satisfiable_with_test(Some(false)) {
            Ok(value) => value,
            Err(reason) => {
                self.error = Some(format!("{}: {reason}", self.file));
                return;
            }
        };
        if reachable_without_test
            && let Some(reason) = crate::source_graph::unclassifiable_macro(mac)
        {
            self.error = Some(format!("{}: {reason}", self.file));
        }
    }
}

fn item_identity(item: &syn::Item, ordinal: usize) -> (&'static str, String, bool) {
    let (kind, name, attrs): (&str, Option<String>, &[Attribute]) = match item {
        syn::Item::Const(item) => ("const", Some(item.ident.to_string()), item.attrs.as_slice()),
        syn::Item::Enum(item) => ("enum", Some(item.ident.to_string()), item.attrs.as_slice()),
        syn::Item::ExternCrate(item) => (
            "extern crate",
            Some(item.ident.to_string()),
            item.attrs.as_slice(),
        ),
        syn::Item::Fn(item) => (
            "function",
            Some(item.sig.ident.to_string()),
            item.attrs.as_slice(),
        ),
        syn::Item::ForeignMod(item) => ("foreign module", None, item.attrs.as_slice()),
        syn::Item::Impl(item) => ("impl", None, item.attrs.as_slice()),
        syn::Item::Macro(item) => (
            "macro",
            item.ident.as_ref().map(ToString::to_string),
            item.attrs.as_slice(),
        ),
        syn::Item::Mod(item) => (
            "module",
            Some(item.ident.to_string()),
            item.attrs.as_slice(),
        ),
        syn::Item::Static(item) => (
            "static",
            Some(item.ident.to_string()),
            item.attrs.as_slice(),
        ),
        syn::Item::Struct(item) => (
            "struct",
            Some(item.ident.to_string()),
            item.attrs.as_slice(),
        ),
        syn::Item::Trait(item) => ("trait", Some(item.ident.to_string()), item.attrs.as_slice()),
        syn::Item::TraitAlias(item) => (
            "trait alias",
            Some(item.ident.to_string()),
            item.attrs.as_slice(),
        ),
        syn::Item::Type(item) => ("type", Some(item.ident.to_string()), item.attrs.as_slice()),
        syn::Item::Union(item) => ("union", Some(item.ident.to_string()), item.attrs.as_slice()),
        syn::Item::Use(item) => ("use", None, item.attrs.as_slice()),
        syn::Item::Verbatim(_) => ("verbatim", None, &[]),
        _ => ("item", None, &[]),
    };
    let name = name.unwrap_or_else(|| format!("{kind}#{ordinal}"));
    (kind, name, has_test_marker(attrs))
}

fn impl_item_identity(
    item: &syn::ImplItem,
    ordinal: usize,
) -> (&[Attribute], &'static str, String, bool) {
    let (attrs, kind, name): (&[Attribute], &str, Option<String>) = match item {
        syn::ImplItem::Const(item) => (&item.attrs, "impl const", Some(item.ident.to_string())),
        syn::ImplItem::Fn(item) => (&item.attrs, "method", Some(item.sig.ident.to_string())),
        syn::ImplItem::Type(item) => (&item.attrs, "impl type", Some(item.ident.to_string())),
        syn::ImplItem::Macro(item) => (&item.attrs, "impl macro", None),
        syn::ImplItem::Verbatim(_) => (&[][..], "impl item", None),
        _ => (&[][..], "impl item", None),
    };
    let name = name.unwrap_or_else(|| format!("{kind}#{ordinal}"));
    (attrs, kind, name, has_test_marker(attrs))
}

fn trait_item_identity(
    item: &syn::TraitItem,
    ordinal: usize,
) -> (&[Attribute], &'static str, String, bool) {
    let (attrs, kind, name): (&[Attribute], &str, Option<String>) = match item {
        syn::TraitItem::Const(item) => (&item.attrs, "trait const", Some(item.ident.to_string())),
        syn::TraitItem::Fn(item) => (
            &item.attrs,
            "trait method",
            Some(item.sig.ident.to_string()),
        ),
        syn::TraitItem::Type(item) => (&item.attrs, "trait type", Some(item.ident.to_string())),
        syn::TraitItem::Macro(item) => (&item.attrs, "trait macro", None),
        syn::TraitItem::Verbatim(_) => (&[][..], "trait item", None),
        _ => (&[][..], "trait item", None),
    };
    let name = name.unwrap_or_else(|| format!("{kind}#{ordinal}"));
    (attrs, kind, name, has_test_marker(attrs))
}

fn foreign_item_identity(
    item: &syn::ForeignItem,
    ordinal: usize,
) -> (&[Attribute], &'static str, String, bool) {
    let (attrs, kind, name): (&[Attribute], &str, Option<String>) = match item {
        syn::ForeignItem::Fn(item) => (
            &item.attrs,
            "foreign function",
            Some(item.sig.ident.to_string()),
        ),
        syn::ForeignItem::Static(item) => {
            (&item.attrs, "foreign static", Some(item.ident.to_string()))
        }
        syn::ForeignItem::Type(item) => (&item.attrs, "foreign type", Some(item.ident.to_string())),
        syn::ForeignItem::Macro(item) => (
            &item.attrs,
            "foreign macro",
            Some(format!("macro#{ordinal}")),
        ),
        syn::ForeignItem::Verbatim(_) => (&[][..], "foreign item", None),
        _ => (&[][..], "foreign item", None),
    };
    let name = name.unwrap_or_else(|| format!("{kind}#{ordinal}"));
    (attrs, kind, name, has_test_marker(attrs))
}

fn has_test_marker(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute
            .path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
    })
}
