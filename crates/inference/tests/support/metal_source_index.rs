use crate::source_graph::{
    CfgFormula, item_attributes, literal_module_path, path_label, syn_attributes_formula_with_test,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use syn::ext::IdentExt;
use syn::spanned::Spanned;

#[derive(Clone)]
pub(super) struct Binding {
    pub target: String,
    pub alias: bool,
    pub cfg: CfgFormula,
}

#[derive(Clone)]
pub(super) struct Function {
    pub module: String,
    pub file: String,
    pub name: String,
    pub self_ty: Option<String>,
    pub trait_path: Option<String>,
    pub cfg: CfgFormula,
    pub definition: Range<usize>,
    pub body: Range<usize>,
    pub signature: String,
    pub public: bool,
    pub inputs: syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
}

#[derive(Clone)]
pub(super) struct TypeInfo {
    pub module: String,
    pub file: String,
    pub cfg: CfgFormula,
    pub fields: BTreeMap<String, String>,
}

pub(super) struct Index {
    pub root: PathBuf,
    pub sources: BTreeMap<String, String>,
    pub file_cfg: BTreeMap<String, CfgFormula>,
    pub bindings: BTreeMap<(String, String), Vec<Binding>>,
    pub globs: BTreeMap<String, Vec<Binding>>,
    pub functions: Vec<Function>,
    pub types: BTreeMap<String, Vec<TypeInfo>>,
    pub module_files: BTreeMap<String, BTreeSet<String>>,
    modules: BTreeSet<String>,
    visiting: BTreeSet<PathBuf>,
}

pub(super) fn intersects(left: &CfgFormula, right: &CfgFormula) -> Result<bool, String> {
    CfgFormula::all([left.clone(), right.clone()]).satisfiable_with_test(Some(false))
}

pub(super) fn type_path(ty: &syn::Type) -> Result<String, String> {
    match ty {
        syn::Type::Path(path) if path.qself.is_none() => Ok(path_label(&path.path)),
        syn::Type::Reference(reference) => type_path(&reference.elem),
        syn::Type::Paren(paren) => type_path(&paren.elem),
        _ => Err("unsupported guarded receiver type".into()),
    }
}

pub(super) fn span_range(source: &str, span: proc_macro2::Span) -> Result<Range<usize>, String> {
    let offset = |point: proc_macro2::LineColumn| -> Result<usize, String> {
        let start = if point.line == 1 {
            0
        } else {
            source
                .match_indices('\n')
                .nth(point.line.saturating_sub(2))
                .map(|(offset, _)| offset + 1)
                .ok_or_else(|| "source span line is outside the file".to_string())?
        };
        let line = source[start..].split('\n').next().unwrap_or("");
        let column = line
            .char_indices()
            .nth(point.column)
            .map(|(offset, _)| offset)
            .or_else(|| (line.chars().count() == point.column).then_some(line.len()))
            .ok_or_else(|| "source span column is outside the line".to_string())?;
        Ok(start + column)
    };
    let range = offset(span.start())?..offset(span.end())?;
    if range.start > range.end || range.end > source.len() {
        return Err("invalid source span".into());
    }
    Ok(range)
}

impl Index {
    pub fn load(root: &Path) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|error| format!("crate root: {error}"))?;
        let mut index = Self {
            root,
            sources: BTreeMap::new(),
            file_cfg: BTreeMap::new(),
            bindings: BTreeMap::new(),
            globs: BTreeMap::new(),
            functions: Vec::new(),
            types: BTreeMap::new(),
            module_files: BTreeMap::new(),
            modules: BTreeSet::new(),
            visiting: BTreeSet::new(),
        };
        let path = index.root.join("src/lib.rs");
        let module_dir = index.root.join("src");
        index.load_file(&path, "crate", &module_dir, CfgFormula::True)?;
        Ok(index)
    }

    fn load_file(
        &mut self,
        path: &Path,
        module: &str,
        module_dir: &Path,
        inherited: CfgFormula,
    ) -> Result<(), String> {
        let path = path
            .canonicalize()
            .map_err(|error| format!("unresolved module {}: {error}", path.display()))?;
        if !path.starts_with(&self.root) {
            return Err(format!("module outside crate: {}", path.display()));
        }
        if !self.visiting.insert(path.clone()) {
            return Err(format!("module cycle at {}", path.display()));
        }
        let file = path
            .strip_prefix(&self.root)
            .map_err(|error| error.to_string())?
            .to_string_lossy()
            .into_owned();
        let source = std::fs::read_to_string(&path).map_err(|error| format!("{file}: {error}"))?;
        let syntax =
            syn::parse_file(&source).map_err(|error| format!("{file}: invalid Rust: {error}"))?;
        let cfg = CfgFormula::all([
            inherited,
            syn_attributes_formula_with_test(&syntax.attrs, None, &file)?,
        ]);
        if cfg.satisfiable_with_test(Some(false))? {
            self.sources.insert(file.clone(), source.clone());
            self.file_cfg
                .entry(file.clone())
                .and_modify(|old| *old = CfgFormula::any([old.clone(), cfg.clone()]))
                .or_insert(cfg.clone());
            self.modules.insert(module.to_string());
            self.module_files
                .entry(module.to_string())
                .or_default()
                .insert(file.clone());
            self.walk(
                &syntax.items,
                module,
                &file,
                &source,
                module_dir,
                path.parent().unwrap_or(module_dir),
                cfg,
            )?;
        }
        self.visiting.remove(&path);
        Ok(())
    }

    fn bind(&mut self, module: &str, name: &str, target: String, alias: bool, cfg: &CfgFormula) {
        self.bindings
            .entry((module.to_string(), name.to_string()))
            .or_default()
            .push(Binding {
                target,
                alias,
                cfg: cfg.clone(),
            });
    }

    fn uses(&mut self, module: &str, prefix: &str, tree: &syn::UseTree, cfg: &CfgFormula) {
        let join = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}::{name}")
            }
        };
        match tree {
            syn::UseTree::Path(path) => self.uses(
                module,
                &join(&path.ident.unraw().to_string()),
                &path.tree,
                cfg,
            ),
            syn::UseTree::Name(name) => {
                let name = name.ident.unraw().to_string();
                if name == "self" {
                    let local = prefix.rsplit("::").next().unwrap_or(prefix);
                    self.bind(module, local, prefix.to_string(), true, cfg);
                } else {
                    self.bind(module, &name, join(&name), true, cfg);
                }
            }
            syn::UseTree::Rename(rename) => {
                let target = if rename.ident == "self" {
                    prefix.to_string()
                } else {
                    join(&rename.ident.unraw().to_string())
                };
                self.bind(
                    module,
                    &rename.rename.unraw().to_string(),
                    target,
                    true,
                    cfg,
                );
            }
            syn::UseTree::Group(group) => {
                for tree in &group.items {
                    self.uses(module, prefix, tree, cfg);
                }
            }
            syn::UseTree::Glob(_) => {
                self.globs
                    .entry(module.to_string())
                    .or_default()
                    .push(Binding {
                        target: prefix.to_string(),
                        alias: true,
                        cfg: cfg.clone(),
                    })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &mut self,
        items: &[syn::Item],
        module: &str,
        file: &str,
        source: &str,
        module_dir: &Path,
        attr_dir: &Path,
        inherited: CfgFormula,
    ) -> Result<(), String> {
        for item in items {
            let cfg = CfgFormula::all([
                inherited.clone(),
                syn_attributes_formula_with_test(item_attributes(item), None, file)?,
            ]);
            if !cfg.satisfiable_with_test(Some(false))?
                || item_attributes(item).iter().any(|attr| {
                    attr.path()
                        .segments
                        .last()
                        .is_some_and(|segment| segment.ident == "test")
                })
            {
                continue;
            }
            match item {
                syn::Item::Mod(child) => {
                    let name = child.ident.unraw().to_string();
                    let child_module = format!("{module}::{name}");
                    self.bind(module, &name, child_module.clone(), false, &cfg);
                    self.modules.insert(child_module.clone());
                    if let Some((_, items)) = &child.content {
                        self.module_files
                            .entry(child_module.clone())
                            .or_default()
                            .insert(file.to_string());
                        if literal_module_path(&child.attrs, file)?.is_some() {
                            return Err(format!("{file}: path on inline module"));
                        }
                        let dir = module_dir.join(&name);
                        self.walk(items, &child_module, file, source, &dir, &dir, cfg)?;
                    } else {
                        let (target, dir) = crate::source_graph::module_destination(
                            child, module_dir, attr_dir, file,
                        )?;
                        self.load_file(&target, &child_module, &dir, cfg)?;
                    }
                }
                syn::Item::Use(item) => self.uses(module, "", &item.tree, &cfg),
                syn::Item::Fn(function) => {
                    let name = function.sig.ident.unraw().to_string();
                    self.bind(module, &name, format!("{module}::{name}"), false, &cfg);
                    self.function(
                        module,
                        file,
                        source,
                        &function.sig,
                        &function.block,
                        function.span(),
                        &function.vis,
                        None,
                        None,
                        cfg,
                    )?;
                }
                syn::Item::Impl(implementation) => {
                    let Ok(self_ty) = type_path(&implementation.self_ty) else {
                        continue;
                    };
                    let trait_path = implementation
                        .trait_
                        .as_ref()
                        .map(|(_, path, _)| path_label(path));
                    for item in &implementation.items {
                        if let syn::ImplItem::Fn(method) = item {
                            let method_cfg = CfgFormula::all([
                                cfg.clone(),
                                syn_attributes_formula_with_test(&method.attrs, None, file)?,
                            ]);
                            let test_marker = method.attrs.iter().any(|attr| {
                                attr.path()
                                    .segments
                                    .last()
                                    .is_some_and(|segment| segment.ident == "test")
                            });
                            if method_cfg.satisfiable_with_test(Some(false))? && !test_marker {
                                self.function(
                                    module,
                                    file,
                                    source,
                                    &method.sig,
                                    &method.block,
                                    method.span(),
                                    &method.vis,
                                    Some(self_ty.clone()),
                                    trait_path.clone(),
                                    method_cfg,
                                )?;
                            }
                        }
                    }
                }
                syn::Item::Struct(item) => {
                    let name = item.ident.unraw().to_string();
                    let identity = format!("{module}::{name}");
                    self.bind(module, &name, identity.clone(), false, &cfg);
                    let fields = item
                        .fields
                        .iter()
                        .enumerate()
                        .filter_map(|(ordinal, field)| {
                            type_path(&field.ty).ok().map(|ty| {
                                (
                                    field
                                        .ident
                                        .as_ref()
                                        .map(|ident| ident.unraw().to_string())
                                        .unwrap_or_else(|| ordinal.to_string()),
                                    ty,
                                )
                            })
                        })
                        .collect();
                    self.types.entry(identity).or_default().push(TypeInfo {
                        module: module.to_string(),
                        file: file.to_string(),
                        cfg,
                        fields,
                    });
                }
                syn::Item::Enum(item) => {
                    let name = item.ident.unraw().to_string();
                    self.bind(module, &name, format!("{module}::{name}"), false, &cfg);
                }
                syn::Item::Trait(item) => {
                    let name = item.ident.unraw().to_string();
                    self.bind(module, &name, format!("{module}::{name}"), false, &cfg);
                }
                syn::Item::Type(item) => {
                    let name = item.ident.unraw().to_string();
                    if let Ok(target) = type_path(&item.ty) {
                        self.bind(module, &name, target, true, &cfg);
                    }
                }
                syn::Item::Macro(item) if item.mac.path.is_ident("include") => {
                    let include = syn::parse2::<syn::LitStr>(item.mac.tokens.clone())
                        .map_err(|error| format!("{file}: unresolved include: {error}"))?;
                    let containing = self.root.join(file);
                    let target = containing
                        .parent()
                        .ok_or_else(|| format!("{file}: no parent"))?
                        .join(include.value());
                    self.load_file(&target, module, module_dir, cfg)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn function(
        &mut self,
        module: &str,
        file: &str,
        source: &str,
        sig: &syn::Signature,
        body: &syn::Block,
        span: proc_macro2::Span,
        vis: &syn::Visibility,
        self_ty: Option<String>,
        trait_path: Option<String>,
        cfg: CfgFormula,
    ) -> Result<(), String> {
        let signature_range = span_range(source, sig.span())?;
        self.functions.push(Function {
            module: module.to_string(),
            file: file.to_string(),
            name: sig.ident.unraw().to_string(),
            self_ty,
            trait_path,
            cfg,
            definition: span_range(source, span)?,
            body: span_range(source, body.span())?,
            signature: source[signature_range].to_string(),
            public: !matches!(vis, syn::Visibility::Inherited),
            inputs: sig.inputs.clone(),
        });
        Ok(())
    }

    pub fn resolve(&self, module: &str, path: &str, cfg: &CfgFormula) -> Result<String, String> {
        self.resolve_inner(module, path, cfg, &mut BTreeSet::new())
    }

    pub fn is_module(&self, path: &str) -> bool {
        self.modules.contains(path)
    }

    fn resolve_inner(
        &self,
        module: &str,
        path: &str,
        cfg: &CfgFormula,
        stack: &mut BTreeSet<String>,
    ) -> Result<String, String> {
        if matches!(path, "std::ops::Drop" | "core::ops::Drop") {
            return Ok("std::ops::Drop".into());
        }
        if path == "Drop"
            && !self
                .bindings
                .contains_key(&(module.to_string(), "Drop".to_string()))
            && !self.globs.contains_key(module)
        {
            return Ok("std::ops::Drop".into());
        }
        let key = format!("{module}|{path}");
        if !stack.insert(key.clone()) {
            return Err(format!("alias cycle resolving {path} in {module}"));
        }
        let result = (|| {
            let mut segments = path
                .split("::")
                .filter(|segment| !segment.is_empty())
                .peekable();
            let first = segments
                .next()
                .ok_or_else(|| "empty symbol path".to_string())?;
            let mut current = match first {
                "crate" => "crate".to_string(),
                "self" => module.to_string(),
                "super" => module
                    .rsplit_once("::")
                    .map(|(parent, _)| parent.to_string())
                    .ok_or_else(|| format!("super escapes root in {path}"))?,
                name => self.lookup(module, name, cfg, stack)?,
            };
            for name in segments {
                if name == "super" {
                    current = current
                        .rsplit_once("::")
                        .map(|(parent, _)| parent.to_string())
                        .ok_or_else(|| format!("super escapes root in {path}"))?;
                } else {
                    current = self.lookup(&current, name, cfg, stack)?;
                }
            }
            Ok(current)
        })();
        stack.remove(&key);
        result
    }

    fn lookup(
        &self,
        module: &str,
        name: &str,
        cfg: &CfgFormula,
        stack: &mut BTreeSet<String>,
    ) -> Result<String, String> {
        let key = format!("lookup:{module}::{name}");
        if !stack.insert(key.clone()) {
            return Err(format!("alias cycle resolving {module}::{name}"));
        }
        let result = (|| {
            let mut answers = BTreeSet::new();
            if let Some(bindings) = self.bindings.get(&(module.to_string(), name.to_string())) {
                for binding in bindings {
                    if intersects(&binding.cfg, cfg)? {
                        let effective = CfgFormula::all([binding.cfg.clone(), cfg.clone()]);
                        answers.insert(if binding.alias {
                            self.resolve_inner(module, &binding.target, &effective, stack)?
                        } else {
                            binding.target.clone()
                        });
                    }
                }
            }
            if answers.is_empty() {
                let mut unresolved = Vec::new();
                for binding in self.globs.get(module).into_iter().flatten() {
                    if !intersects(&binding.cfg, cfg)? {
                        continue;
                    }
                    let effective = CfgFormula::all([binding.cfg.clone(), cfg.clone()]);
                    match self.resolve_inner(module, &binding.target, &effective, stack) {
                        Ok(destination) => {
                            if self
                                .bindings
                                .contains_key(&(destination.clone(), name.to_string()))
                                || self.globs.contains_key(&destination)
                            {
                                answers.insert(self.lookup(
                                    &destination,
                                    name,
                                    &effective,
                                    stack,
                                )?);
                            }
                        }
                        Err(error) => unresolved.push(error),
                    }
                }
                if !unresolved.is_empty() {
                    return Err(format!(
                        "unresolved ownership of {module}::{name}: {}",
                        unresolved.join("; ")
                    ));
                }
            }
            match answers.len() {
                1 => answers
                    .into_iter()
                    .next()
                    .ok_or_else(|| "missing resolved symbol".into()),
                0 => Err(format!("unresolved symbol {module}::{name}")),
                _ => Err(format!(
                    "ambiguous ownership of {module}::{name}: {answers:?}"
                )),
            }
        })();
        stack.remove(&key);
        result
    }

    pub fn function_identity(
        &self,
        function: &Function,
        cfg: &CfgFormula,
    ) -> Result<String, String> {
        if let Some(self_ty) = &function.self_ty {
            let owner = self.resolve(&function.module, self_ty, cfg)?;
            if let Some(trait_path) = &function.trait_path {
                let trait_name = self.resolve(&function.module, trait_path, cfg)?;
                Ok(format!("<{owner} as {trait_name}>::{}", function.name))
            } else {
                Ok(format!("{owner}::{}", function.name))
            }
        } else {
            Ok(format!("{}::{}", function.module, function.name))
        }
    }

    pub fn field_type(&self, owner: &str, field: &str, cfg: &CfgFormula) -> Result<String, String> {
        let mut answers = BTreeSet::new();
        for info in self.types.get(owner).into_iter().flatten() {
            if intersects(&info.cfg, cfg)? {
                let ty = info
                    .fields
                    .get(field)
                    .ok_or_else(|| format!("unsupported field {owner}.{field} in {}", info.file))?;
                answers.insert(self.resolve(&info.module, ty, cfg)?);
            }
        }
        if answers.len() != 1 {
            return Err(format!("unresolved or ambiguous receiver {owner}.{field}"));
        }
        answers
            .into_iter()
            .next()
            .ok_or_else(|| "missing field receiver".into())
    }
}
