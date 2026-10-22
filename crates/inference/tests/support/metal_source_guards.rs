//! Resolved owners and production source views for the Metal source contracts.

#[path = "metal_source_index.rs"]
mod index;
#[path = "metal_source_view.rs"]
mod view;

use crate::metal_qwen35_population::{classify_production_items, derive_population};
use crate::source_graph::{CfgFormula, module_source_population_closure, path_label};
use index::{Function, Index, intersects, span_range};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::Path;
use syn::ext::IdentExt;
use syn::spanned::Spanned;
use syn::visit::Visit;

#[derive(Clone, Copy)]
pub(crate) enum Scope {
    Production,
    Metal,
}

impl Scope {
    fn cfg(self) -> CfgFormula {
        match self {
            Self::Production => CfgFormula::True,
            Self::Metal => CfgFormula::all([
                CfgFormula::Atom("target_os=macos".into()),
                CfgFormula::Atom("feature=metal-gpu".into()),
            ]),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum OwnerKind {
    Function,
    InherentMethod {
        self_ty: &'static str,
    },
    TraitMethod {
        self_ty: &'static str,
        trait_path: &'static str,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct OwnerSpec {
    pub(crate) id: &'static str,
    pub(crate) path: &'static str,
    pub(crate) kind: OwnerKind,
    pub(crate) scope: Scope,
}

pub(crate) struct CallSpec {
    pub(crate) caller: &'static str,
    pub(crate) callee: &'static str,
    pub(crate) count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedOwner {
    pub(crate) file: String,
    pub(crate) definition_range: Range<usize>,
    pub(crate) body_range: Range<usize>,
    pub(crate) semantic_identity: String,
    pub(crate) signature: String,
    pub(crate) is_public: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedCall {
    pub(crate) caller: String,
    pub(crate) file: String,
    pub(crate) range: Range<usize>,
}

pub(crate) struct GuardPopulation {
    index: Index,
    sources: BTreeMap<String, String>,
    owners: BTreeMap<String, ResolvedOwner>,
    specs: BTreeMap<String, OwnerSpec>,
}

impl GuardPopulation {
    pub(crate) fn load(crate_dir: &Path, owners: &[OwnerSpec]) -> Result<Self, String> {
        Self::load_with_discovery_filter(crate_dir, owners, |_| true)
    }

    pub(crate) fn load_with_discovery_filter(
        crate_dir: &Path,
        owners: &[OwnerSpec],
        keep: impl Fn(&str) -> bool,
    ) -> Result<Self, String> {
        Self::prepare(crate_dir, owners, |required| {
            Ok(required.iter().filter(|path| keep(path)).cloned().collect())
        })
    }

    pub(crate) fn load_with_candidates(
        crate_dir: &Path,
        owners: &[OwnerSpec],
        candidates: &[String],
    ) -> Result<Self, String> {
        Self::prepare(crate_dir, owners, |_| Ok(candidates.to_vec()))
    }

    fn prepare(
        crate_dir: &Path,
        specs: &[OwnerSpec],
        discover: impl FnOnce(&BTreeSet<String>) -> Result<Vec<String>, String>,
    ) -> Result<Self, String> {
        if specs.is_empty() {
            return Err("owner manifest is empty".into());
        }
        let index = Index::load(crate_dir)?;
        let mut owners = BTreeMap::new();
        let mut spec_map = BTreeMap::new();
        let mut resolved_functions = Vec::new();
        for spec in specs {
            if spec_map.insert(spec.id.to_string(), *spec).is_some() {
                return Err(format!("duplicate owner obligation {}", spec.id));
            }
            let function = resolve_owner(&index, spec)?;
            let identity = index.function_identity(
                function,
                &CfgFormula::all([spec.scope.cfg(), function.cfg.clone()]),
            )?;
            owners.insert(
                spec.id.to_string(),
                ResolvedOwner {
                    file: function.file.clone(),
                    definition_range: function.definition.clone(),
                    body_range: function.body.clone(),
                    semantic_identity: identity,
                    signature: function.signature.clone(),
                    is_public: function.public,
                },
            );
            resolved_functions.push(function);
        }

        let population = derive_population(
            crate_dir,
            Path::new("src/forward/metal_qwen35.rs"),
            Path::new("src/forward/metal_qwen35"),
        )?;
        let mut required = population
            .discovered_files
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let decoder = crate_dir.join("src/decoder/qwen_metal.rs");
        let decoder_dir = crate_dir.join("src/decoder/qwen_metal");
        for source in module_source_population_closure(crate_dir, &decoder, &decoder_dir)? {
            required.insert(
                source
                    .path
                    .strip_prefix(&index.root)
                    .map_err(|error| error.to_string())?
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        for function in &resolved_functions {
            required.insert(function.file.clone());
            for (module, files) in &index.module_files {
                if module == &function.module
                    || module.starts_with(&format!("{}::", function.module))
                {
                    required.extend(files.iter().cloned());
                }
            }
        }
        let candidates = discover(&required)?;
        let mut discovered = BTreeSet::new();
        for file in candidates {
            if !discovered.insert(file.clone()) {
                return Err(format!("duplicate discovered source {file}"));
            }
        }
        let missing = required
            .difference(&discovered)
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "resolved production source coverage is incomplete; missing {}",
                missing.join(", ")
            ));
        }
        let extras = discovered
            .difference(&required)
            .cloned()
            .collect::<Vec<_>>();
        if !extras.is_empty() {
            return Err(format!(
                "unclassified extra discovered sources: {}",
                extras.join(", ")
            ));
        }
        let mut sources = BTreeMap::new();
        let mut read_items = Vec::new();
        for file in &discovered {
            if let Some(source) = index.sources.get(file) {
                let cfg = index
                    .file_cfg
                    .get(file)
                    .ok_or_else(|| format!("missing source cfg for {file}"))?
                    .clone();
                read_items.extend(classify_production_items(file, source, cfg.clone())?);
                sources.insert(file.clone(), view::production_source(file, source, cfg)?);
            } else {
                let path = index.root.join(file);
                let source = std::fs::read_to_string(&path)
                    .map_err(|error| format!("{file}: unread source: {error}"))?;
                let empty = source
                    .bytes()
                    .map(|byte| {
                        if matches!(byte, b'\n' | b'\r') {
                            byte
                        } else {
                            b' '
                        }
                    })
                    .map(char::from)
                    .collect();
                // Test-only files are still read; no production item is fabricated for them.
                sources.insert(file.clone(), empty);
            }
        }
        population.require_production_items_read(&read_items)?;
        Ok(Self {
            index,
            sources,
            owners,
            specs: spec_map,
        })
    }

    pub(crate) fn owner(&self, id: &str) -> Result<&ResolvedOwner, String> {
        self.owners
            .get(id)
            .ok_or_else(|| format!("undeclared owner {id}"))
    }

    pub(crate) fn owner_body(&self, id: &str) -> Result<&str, String> {
        let owner = self.owner(id)?;
        self.sources
            .get(&owner.file)
            .and_then(|source| source.get(owner.body_range.clone()))
            .ok_or_else(|| format!("owner {id} has no classified body"))
    }

    pub(crate) fn production_sources(&self) -> impl Iterator<Item = (&str, &str)> {
        self.sources
            .iter()
            .map(|(path, source)| (path.as_str(), source.as_str()))
    }

    pub(crate) fn calls_to(&self, callee_id: &str) -> Result<Vec<ResolvedCall>, String> {
        self.calls_in(callee_id, None)
    }

    fn calls_in(
        &self,
        callee_id: &str,
        only_caller: Option<&str>,
    ) -> Result<Vec<ResolvedCall>, String> {
        let callee = self.owner(callee_id)?;
        let spec = self
            .specs
            .get(callee_id)
            .ok_or_else(|| format!("undeclared callee {callee_id}"))?;
        let mut calls = Vec::new();
        for function in &self.index.functions {
            let Some(source) = self.sources.get(&function.file) else {
                continue;
            };
            let cfg = CfgFormula::all([function.cfg.clone(), spec.scope.cfg()]);
            if !cfg.satisfiable_with_test(Some(false))? {
                continue;
            }
            if let Some(caller) = only_caller
                && self.index.function_identity(function, &cfg).as_deref() != Ok(caller)
            {
                continue;
            }
            let body = source
                .get(function.body.clone())
                .ok_or_else(|| format!("invalid body span in {}", function.file))?;
            if body.trim().is_empty() {
                continue;
            }
            let block = syn::parse_str::<syn::Block>(body)
                .map_err(|error| format!("{}: classified body: {error}", function.file))?;
            let mut collector = Calls {
                index: &self.index,
                function,
                callee: &callee.semantic_identity,
                cfg,
                body,
                found: Vec::new(),
                error: None,
                shadows: BTreeSet::new(),
                local_aliases: BTreeMap::new(),
                local_globs: Vec::new(),
                global_census: only_caller.is_none(),
            };
            for input in &function.inputs {
                if let syn::FnArg::Typed(argument) = input {
                    collect_patterns(&argument.pat, &mut collector.shadows);
                }
            }
            collector.visit_block(&block);
            if let Some(error) = collector.error {
                return Err(format!("callee {callee_id}: {error}"));
            }
            if !collector.found.is_empty() {
                let caller = self.index.function_identity(function, &collector.cfg)?;
                for range in collector.found {
                    calls.push(ResolvedCall {
                        caller: caller.clone(),
                        file: function.file.clone(),
                        range: (function.body.start + range.start)
                            ..(function.body.start + range.end),
                    });
                }
            }
        }
        calls.sort_by(|left, right| {
            (&left.file, left.range.start, &left.caller).cmp(&(
                &right.file,
                right.range.start,
                &right.caller,
            ))
        });
        calls.dedup();
        Ok(calls)
    }

    pub(crate) fn call_sites(
        &self,
        caller_id: &str,
        callee_id: &str,
    ) -> Result<Vec<ResolvedCall>, String> {
        let caller = &self.owner(caller_id)?.semantic_identity;
        self.calls_in(callee_id, Some(caller))
    }

    pub(crate) fn require_calls(&self, expected: &[CallSpec]) -> Result<(), String> {
        let mut expected_by_callee: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
        for call in expected {
            let caller = self.owner(call.caller)?.semantic_identity.clone();
            self.owner(call.callee)?;
            if expected_by_callee
                .entry(call.callee)
                .or_default()
                .insert(caller, call.count)
                .is_some()
            {
                return Err(format!(
                    "duplicate caller obligation {} -> {}",
                    call.caller, call.callee
                ));
            }
        }
        for (callee, expected) in expected_by_callee {
            let mut actual = BTreeMap::new();
            for call in self.calls_to(callee)? {
                *actual.entry(call.caller).or_insert(0usize) += 1;
            }
            let expected = expected
                .into_iter()
                .filter(|(_, count)| *count != 0)
                .collect::<BTreeMap<_, _>>();
            if actual != expected {
                return Err(format!(
                    "callee {callee} caller manifest differs: expected {expected:?}, found {actual:?}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn reject_declared_functions(&self, names: &[&str]) -> Result<(), String> {
        for (file, source) in &self.sources {
            let syntax = syn::parse_file(source).map_err(|error| format!("{file}: {error}"))?;
            let mut declarations = Declarations { names, found: None };
            declarations.visit_file(&syntax);
            if let Some(name) = declarations.found {
                return Err(format!("retired declaration {name} in {file}"));
            }
        }
        Ok(())
    }
}

fn resolve_owner<'a>(index: &'a Index, spec: &OwnerSpec) -> Result<&'a Function, String> {
    let cfg = spec.scope.cfg();
    let expected = match spec.kind {
        OwnerKind::Function => index.resolve("crate", spec.path, &cfg)?,
        OwnerKind::InherentMethod { self_ty } => {
            format!("{}::{}", index.resolve("crate", self_ty, &cfg)?, spec.path)
        }
        OwnerKind::TraitMethod {
            self_ty,
            trait_path,
        } => format!(
            "<{} as {}>::{}",
            index.resolve("crate", self_ty, &cfg)?,
            index.resolve("crate", trait_path, &cfg)?,
            spec.path
        ),
    };
    let leaf = spec.path.rsplit("::").next().unwrap_or(spec.path);
    let mut answers = Vec::new();
    for function in &index.functions {
        if function.name != leaf || !intersects(&function.cfg, &cfg)? {
            continue;
        }
        let matching_kind = match spec.kind {
            OwnerKind::Function => function.self_ty.is_none(),
            OwnerKind::InherentMethod { .. } => {
                function.self_ty.is_some() && function.trait_path.is_none()
            }
            OwnerKind::TraitMethod { .. } => function.trait_path.is_some(),
        };
        if !matching_kind {
            continue;
        }
        let active = CfgFormula::all([cfg.clone(), function.cfg.clone()]);
        if let Ok(identity) = index.function_identity(function, &active)
            && identity == expected
        {
            answers.push(function);
        }
    }
    answers.sort_by_key(|function| (&function.file, function.definition.start));
    answers.dedup_by(|left, right| left.file == right.file && left.definition == right.definition);
    match answers.len() {
        1 => Ok(answers[0]),
        0 => Err(format!("unresolved owner {}: {expected}", spec.id)),
        _ => Err(format!("ambiguous owner {}: {expected}", spec.id)),
    }
}

struct Declarations<'a> {
    names: &'a [&'a str],
    found: Option<String>,
}
impl<'ast> Visit<'ast> for Declarations<'_> {
    fn visit_signature(&mut self, signature: &'ast syn::Signature) {
        let name = signature.ident.unraw().to_string();
        if self.names.contains(&name.as_str()) {
            self.found = Some(name);
        }
        syn::visit::visit_signature(self, signature);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        fn declaration(tokens: proc_macro2::TokenStream, names: &[&str]) -> Option<String> {
            let mut after_fn = false;
            for token in tokens {
                match token {
                    proc_macro2::TokenTree::Ident(ident) => {
                        let name = ident.unraw().to_string();
                        if after_fn && names.contains(&name.as_str()) {
                            return Some(name);
                        }
                        after_fn = name == "fn";
                    }
                    proc_macro2::TokenTree::Group(group) => {
                        if let Some(name) = declaration(group.stream(), names) {
                            return Some(name);
                        }
                        after_fn = false;
                    }
                    _ => after_fn = false,
                }
            }
            None
        }
        if let Some(name) = declaration(mac.tokens.clone(), self.names) {
            self.found = Some(name);
        }
    }
}

fn collect_patterns(pattern: &syn::Pat, names: &mut BTreeSet<String>) {
    struct Names<'a>(&'a mut BTreeSet<String>);
    impl<'ast> Visit<'ast> for Names<'_> {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            self.0.insert(pattern.ident.unraw().to_string());
            syn::visit::visit_pat_ident(self, pattern);
        }
    }
    Names(names).visit_pat(pattern);
}

struct Calls<'a> {
    index: &'a Index,
    function: &'a Function,
    callee: &'a str,
    cfg: CfgFormula,
    body: &'a str,
    found: Vec<Range<usize>>,
    error: Option<String>,
    shadows: BTreeSet<String>,
    local_aliases: BTreeMap<String, String>,
    local_globs: Vec<String>,
    global_census: bool,
}

impl Calls<'_> {
    fn resolve_path(&self, path: &syn::Path) -> Result<String, String> {
        let label = path_label(path);
        let first = label.split("::").next().unwrap_or("");
        if self.shadows.contains(first) {
            return Err(format!("local binding shadows guarded call {label}"));
        }
        let use_globs = !label.contains("::") && !self.local_aliases.contains_key(first);
        let label = if first == "Self" {
            let self_ty = self
                .function
                .self_ty
                .as_ref()
                .ok_or_else(|| "Self outside an impl".to_string())?;
            let owner = self
                .index
                .resolve(&self.function.module, self_ty, &self.cfg)?;
            format!("{owner}{}", &label[first.len()..])
        } else if let Some(target) = self.local_aliases.get(first) {
            format!("{target}{}", &label[first.len()..])
        } else {
            label
        };
        let mut glob_answers = BTreeSet::new();
        for glob in self.local_globs.iter().filter(|_| use_globs) {
            let owner = self.index.resolve(&self.function.module, glob, &self.cfg)?;
            if let Ok(target) = self.index.resolve(
                &self.function.module,
                &format!("{owner}::{label}"),
                &self.cfg,
            ) {
                glob_answers.insert(target);
            }
        }
        if glob_answers.len() > 1 {
            return Err(format!("ambiguous local glob ownership for {label}"));
        }
        if let Some(target) = glob_answers.into_iter().next() {
            return Ok(target);
        }
        if let Ok(resolved) = self.index.resolve(&self.function.module, &label, &self.cfg) {
            return Ok(resolved);
        }
        if let Some((owner, method)) = label.rsplit_once("::")
            && let Ok(owner) = self.index.resolve(&self.function.module, owner, &self.cfg)
            && !self.index.is_module(&owner)
        {
            return Ok(format!("{owner}::{method}"));
        }
        self.index.resolve(&self.function.module, &label, &self.cfg)
    }

    fn receiver_type(&self, expr: &syn::Expr) -> Result<String, String> {
        match expr {
            syn::Expr::Path(path) if path.path.is_ident("self") => self
                .function
                .self_ty
                .as_ref()
                .ok_or_else(|| "self outside method".to_string())
                .and_then(|ty| self.index.resolve(&self.function.module, ty, &self.cfg)),
            syn::Expr::Field(field) => {
                let owner = self.receiver_type(&field.base)?;
                let member = match &field.member {
                    syn::Member::Named(name) => name.unraw().to_string(),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                };
                self.index.field_type(&owner, &member, &self.cfg)
            }
            syn::Expr::Paren(paren) => self.receiver_type(&paren.expr),
            syn::Expr::Reference(reference) => self.receiver_type(&reference.expr),
            _ => Err("unsupported guarded method receiver".into()),
        }
    }

    fn relevant_leaf(&self, name: &str) -> bool {
        self.callee.rsplit("::").next() == Some(name)
    }

    fn unresolved_relevant(&self, path: &syn::Path) -> bool {
        let label = path_label(path);
        let Some((prefix, leaf)) = label.rsplit_once("::") else {
            return self.relevant_leaf(&label) || self.local_aliases.contains_key(&label);
        };
        if !self.relevant_leaf(leaf) {
            return false;
        }
        if self.global_census {
            return true;
        }
        let first = prefix.split("::").next().unwrap_or("");
        if first == "Self" || self.local_aliases.contains_key(first) {
            return true;
        }
        self.index
            .resolve(&self.function.module, prefix, &self.cfg)
            .is_ok_and(|owner| {
                self.callee == format!("{owner}::{leaf}") || self.index.is_module(&owner)
            })
    }

    fn record(&mut self, span: proc_macro2::Span) {
        match span_range(self.body, span) {
            Ok(range) => self.found.push(range),
            Err(error) => self.error = Some(error),
        }
    }
}

impl<'ast> Visit<'ast> for Calls<'_> {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        let shadows = self.shadows.clone();
        let aliases = self.local_aliases.clone();
        let globs = self.local_globs.clone();
        for statement in &block.stmts {
            if let syn::Stmt::Item(item @ syn::Item::Use(_)) = statement {
                self.visit_item(item);
            } else if let syn::Stmt::Item(syn::Item::Fn(function)) = statement {
                self.shadows.insert(function.sig.ident.unraw().to_string());
            }
        }
        for statement in &block.stmts {
            if !matches!(statement, syn::Stmt::Item(syn::Item::Use(_))) {
                self.visit_stmt(statement);
            }
        }
        self.shadows = shadows;
        self.local_aliases = aliases;
        self.local_globs = globs;
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if self.error.is_some() {
            return;
        }
        if let syn::Expr::Path(path) = call.func.as_ref() {
            if path.qself.is_some() && self.unresolved_relevant(&path.path) {
                self.error = Some(format!(
                    "{}: qualified trait call needs an explicit resolution rule",
                    self.function.file
                ));
                return;
            }
            match self.resolve_path(&path.path) {
                Ok(target) if target == self.callee => self.record(call.span()),
                Err(error) if self.unresolved_relevant(&path.path) => {
                    self.error = Some(format!("{}: {error}", self.function.file))
                }
                _ => {}
            }
        } else {
            syn::visit::visit_expr(self, &call.func);
        }
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let name = call.method.unraw().to_string();
        if self.relevant_leaf(&name) {
            match self.receiver_type(&call.receiver) {
                Ok(owner) if format!("{owner}::{name}") == self.callee => self.record(call.span()),
                Err(error) => self.error = Some(format!("{}: {name}: {error}", self.function.file)),
                _ => {}
            }
        }
        self.visit_expr(&call.receiver);
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if self
            .resolve_path(&path.path)
            .is_ok_and(|target| target == self.callee)
        {
            self.error = Some(format!(
                "{}: guarded function {} is used through an unsupported value alias",
                self.function.file, self.callee
            ));
        }
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        syn::visit::visit_local(self, local);
        collect_patterns(&local.pat, &mut self.shadows);
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        let shadows = self.shadows.clone();
        for pattern in &closure.inputs {
            collect_patterns(pattern, &mut self.shadows);
        }
        self.visit_expr(&closure.body);
        self.shadows = shadows;
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        self.visit_expr(&expr.expr);
        let shadows = self.shadows.clone();
        collect_patterns(&expr.pat, &mut self.shadows);
        self.visit_block(&expr.body);
        self.shadows = shadows;
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        let shadows = self.shadows.clone();
        self.visit_expr(&expr.cond);
        self.visit_block(&expr.then_branch);
        self.shadows = shadows;
        if let Some((_, otherwise)) = &expr.else_branch {
            self.visit_expr(otherwise);
        }
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        let shadows = self.shadows.clone();
        self.visit_expr(&expr.cond);
        self.visit_block(&expr.body);
        self.shadows = shadows;
    }

    fn visit_expr_let(&mut self, expr: &'ast syn::ExprLet) {
        self.visit_expr(&expr.expr);
        collect_patterns(&expr.pat, &mut self.shadows);
    }

    fn visit_arm(&mut self, arm: &'ast syn::Arm) {
        let shadows = self.shadows.clone();
        collect_patterns(&arm.pat, &mut self.shadows);
        if let Some((_, guard)) = &arm.guard {
            self.visit_expr(guard);
        }
        self.visit_expr(&arm.body);
        self.shadows = shadows;
    }

    fn visit_item(&mut self, item: &'ast syn::Item) {
        if let syn::Item::Use(item) = item {
            fn flatten(
                prefix: &str,
                tree: &syn::UseTree,
                result: &mut BTreeMap<String, String>,
                globs: &mut Vec<String>,
            ) -> Result<(), String> {
                let join = |name: &str| {
                    if prefix.is_empty() {
                        name.to_string()
                    } else {
                        format!("{prefix}::{name}")
                    }
                };
                match tree {
                    syn::UseTree::Path(path) => flatten(
                        &join(&path.ident.unraw().to_string()),
                        &path.tree,
                        result,
                        globs,
                    ),
                    syn::UseTree::Name(name) => {
                        let name = name.ident.unraw().to_string();
                        let (local, target) = if name == "self" {
                            (
                                prefix.rsplit("::").next().unwrap_or(prefix).to_string(),
                                prefix.to_string(),
                            )
                        } else {
                            (name.clone(), join(&name))
                        };
                        result.insert(local, target);
                        Ok(())
                    }
                    syn::UseTree::Rename(rename) => {
                        let target = if rename.ident == "self" {
                            prefix.to_string()
                        } else {
                            join(&rename.ident.unraw().to_string())
                        };
                        result.insert(rename.rename.unraw().to_string(), target);
                        Ok(())
                    }
                    syn::UseTree::Group(group) => {
                        for tree in &group.items {
                            flatten(prefix, tree, result, globs)?;
                        }
                        Ok(())
                    }
                    syn::UseTree::Glob(_) => {
                        globs.push(prefix.to_string());
                        Ok(())
                    }
                }
            }
            if let Err(error) = flatten(
                "",
                &item.tree,
                &mut self.local_aliases,
                &mut self.local_globs,
            ) {
                self.error = Some(format!("{}: {error}", self.function.file));
            }
        } else if let syn::Item::Fn(function) = item {
            let previous = self.found.len();
            syn::visit::visit_item_fn(self, function);
            if self.found.len() != previous {
                self.error = Some(format!(
                    "{}: guarded call in a nested function needs an explicit owner",
                    self.function.file
                ));
            }
        } else {
            let previous = self.found.len();
            syn::visit::visit_item(self, item);
            if self.found.len() != previous {
                self.error = Some(format!(
                    "{}: guarded call in a nested item needs an explicit owner",
                    self.function.file
                ));
            }
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        let leaf = self.callee.rsplit("::").next().unwrap_or("");
        fn contains(
            tokens: proc_macro2::TokenStream,
            predicate: &impl Fn(proc_macro2::Ident) -> bool,
        ) -> bool {
            tokens.into_iter().any(|token| match token {
                proc_macro2::TokenTree::Ident(ident) => predicate(ident),
                proc_macro2::TokenTree::Group(group) => contains(group.stream(), predicate),
                _ => false,
            })
        }
        if contains(mac.tokens.clone(), &|ident| {
            ident == leaf
                || self
                    .resolve_path(&syn::Path::from(ident))
                    .is_ok_and(|target| target == self.callee)
        }) {
            self.error = Some(format!(
                "{}: guarded call {leaf} is hidden in a macro",
                self.function.file
            ));
        }
    }
}
