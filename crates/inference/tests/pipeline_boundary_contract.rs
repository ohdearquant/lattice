//! ADR-092 "Completion requires both dependency and execution evidence" —
//! the source-level `pipeline_boundary_contract` resolver.
//!
//! This is a bounded, source-only dependency scan of the production
//! frontend roots (`src/serve`, the provider files, `src/serving_worker.rs`,
//! and every `src/bin` target) for concrete model/config/backend-state dependencies. It does
//! NOT establish numerical or device execution, and it does not itself
//! migrate anything: it is a ratchet against a recorded baseline, run in
//! two configurations (default features, and `metal-gpu,f16` on macOS).
//!
//! ## Origin rule (residual risk, stated up front)
//!
//! An item is "origin" (concrete model/config/backend-state) iff the file
//! that DEFINES it lives under `crate_dir/model/` or matches
//! `crate_dir/forward/metal_*` (a directory-name prefix match on the
//! canonical path, which covers nested trees such as
//! `forward/metal_qwen35/inner/...` and `forward/metal_ernie45/state/...`
//! for free, and excludes `forward/metal.rs`, which has no trailing
//! underscore). Everything else under `forward/` — CPU kernels, pooling,
//! GEMM dispatch — is treated as a reviewed neutral operation with opaque
//! storage. This is the narrowing most likely to be wrong and is called out
//! here rather than left implicit, per the ADR's own instruction.
//!
//! ## Namespace model (an interpretation, not something the ADR states)
//!
//! `analyze_boundary`'s `frontend_roots`/`index_roots` are bare file paths;
//! nothing in the ADR text says how a root's own `crate`/`self`/`super`
//! addressing should be anchored. This resolver anchors it as follows,
//! because `src/bin/*.rs` files reference the library crate 393 times (by
//! grep, at the time this was written) via the literal prefix
//! `lattice_inference::`, never via `crate::` — confirming that bin-to-lib
//! crossing is the dominant real pattern, not an edge case to special-case
//! away:
//!
//! - A root that IS a known Cargo target (the `lib` target, or a `bin`
//!   target) gets its own namespace and an empty starting module path.
//!   The `lib` target's namespace is additionally reachable, from any
//!   other namespace, via the crate's own package name (read from
//!   `Cargo.toml`, `-` mapped to `_`) as if it were a declared extern
//!   crate — because from a `bin` target's perspective, the library really
//!   is an external crate.
//! - A root that is NOT itself a Cargo target (`src/serve/mod.rs`, passed
//!   as a frontend root) is looked up in the SAME namespace's already
//!   walked files if it was reached from an index root; if not, it starts
//!   a starting module path inferred from its own file name (the parent
//!   directory name if the file is literally `mod.rs`, else the file
//!   stem) — which is how `src/serve/mod.rs` acquires the module path
//!   `["serve"]` instead of being (wrongly) treated as its own crate root.
//! - A frontend root's scanned file set is every file in its namespace
//!   whose module path *starts with* that root's own module path — a
//!   module-graph membership test, not a directory-prefix test, which is
//!   what lets a `#[path]`-relocated sibling reached from inside `serve/`
//!   still count as frontend even though it physically lives outside
//!   `src/serve/`.
//! - Fixture crates (no `Cargo.toml` at `crate_dir`) collapse to one
//!   shared namespace, so a fixture can hand `analyze_boundary` a
//!   `model/` root and a `frontend.rs` root separately without a
//!   connecting `lib.rs`.
//!
//! ## Resolution scope (stated, not left implicit)
//!
//! References are collected from the syn AST: `use` trees, type positions,
//! expression positions, and one bounded local-inference rule for
//! `let x = f(..)` where `f` is a direct path to an indexed function with a
//! directly-named return type. A resolved absolute path is looked up by
//! prefix, longest first, against a definitions/re-exports index built
//! from the SAME kinds this scan itself needs to classify (struct, enum,
//! union, trait, type alias, top-level `const`/`static`, and top-level
//! `fn`, so that a reference to an ordinary neutral function or constant
//! such as `crate::forward::cpu_f16::embed_text_vlm_f16` resolves to that
//! item's own (non-origin) file instead of falling through to unresolved).
//! A `pub use` re-export is chased up to depth 8; exceeding it is
//! unresolved. `std`/`core`/`alloc` and any extern crate reached through a
//! module-local `use` (declared dependency or not) resolve through that
//! `use`'s own alias entry to a dedicated external sentinel, so that a
//! LATER multi-segment reference through the imported name (`use
//! std::path::PathBuf; ... PathBuf::from(x)`, `use tokio::sync::mpsc; ...
//! mpsc::unbounded_channel()`) is external too, not merely the `use` line
//! itself -- a single-segment name has no such alias-independent path
//! (see the "local binding" rule below), which is what makes this an
//! alias-table concern rather than a first-segment string match.
//!
//! A single-segment identifier that resolves to nothing (not an alias, not
//! a definition, not an external/known name) is treated as a local binding
//! — a function parameter, a `let` variable, a generic parameter, `Self`
//! with no resolvable concrete self type, or a prelude/primitive name —
//! and is NOT added to `unresolved`. This is deliberate and is what keeps
//! ordinary Rust from flooding the completeness check: a local binding can
//! never itself be a multi-segment path in valid Rust, so this carve-out
//! cannot mask a real multi-segment crossing. A MULTI-segment path that
//! resolves to nothing is unresolved, per the ADR's own "unresolved name or
//! ambiguous classification must fail completeness" instruction. This is a
//! narrower reading of "local binding" than the ADR states in one sentence
//! without defining it, and it is the second-most-likely-wrong narrowing
//! after the origin-directory rule above.
//!
//! `use` trees follow Rust 2018+ import-path rules specifically: an
//! unqualified first segment of a `use` tree (not `crate`/`self`/`super`)
//! always names an extern crate (including the current crate under its own
//! name), never a same-crate sibling module reached by its bare name. An
//! unrecognized such name is treated as an external crate this resolver
//! did not enumerate (a dev/optional dependency, say) rather than as
//! unresolved, which is a deliberately permissive interior default — it
//! narrows the ALIAS TABLE, not the origin classification of anything
//! actually reached from `model/` or `forward/metal_*`, which is decided
//! purely by defining-file path regardless of how it was imported.
//!
//! Unknown/unclassifiable macro expansions (including `include!`, at both
//! item and expression position) inside a scanned file are a completeness
//! failure, reusing `unclassifiable_macro` / `unclassifiable_item_macro` /
//! `unclassifiable_include` from the shared source-analysis module
//! verbatim.
//!
//! ## Third narrowing: generic type parameters and associated items
//!
//! A path headed by an in-scope generic type parameter (`D::Error`,
//! `A::Error::custom` where `D`/`A` is a type parameter of the enclosing
//! `impl`, `fn`, or method) is treated as a local binding, the same
//! reasoning as the single-segment carve-out above, via a `generic_scopes`
//! stack (`generic_type_names`, pushed/popped per `impl`/`fn`/method).
//! Without it, a bound-but-unresolvable-by-name type parameter would read
//! as a multi-segment path to nowhere and be a false completeness failure.
//! A DIFFERENT case that reads similarly but is not this one:
//! `Self::Value`-style associated-type paths, where `Value` is an
//! associated type the SAME `impl` block declares (`impl<'de> Visitor<'de>
//! for X { type Value = ...; }`), not a generic parameter of that impl.
//! This is resolved by a separate, per-impl `assoc_type_scopes` map
//! (`assoc_type_targets`), checked before the ordinary `Self`
//! substitution, specifically so two different `impl` blocks for the same
//! concrete type can each give `Value` a different meaning without
//! collision. Both carve-outs are scoped by a stack tied to the enclosing
//! `impl`/`fn`, never looked up globally by name -- the residual risk they
//! share with the fourth narrowing below is that none of these stacks can
//! see past its own declared scope.
//!
//! ## Fourth narrowing: block-scoped local items
//!
//! A `struct`/`enum`/`trait`/`fn`/etc. declared physically inside a
//! function body (legal Rust: `fn get_or_compile() { enum Action { ... } }`)
//! is invisible to this resolver's item index -- the index only descends
//! into inline `mod name { ... }` blocks, never function bodies -- so a
//! reference to such an item's bare name would otherwise be a false
//! completeness failure. `LocalItemNames` (a lightweight `syn::Visit` that
//! collects bare item NAMES, not paths, from anywhere in a file via
//! unmodified default recursion) is consulted as the LAST resort, after
//! every other resolution path has failed, on the invariant that every
//! production frontend roots are `src/serve`, the provider files
//! `src/serving_provider/providers/{bert.rs,bert/embedding.rs,gemma.rs,gemma/worker.rs,qwen.rs,qwen/adapters.rs,qwen/embedding.rs,qwen/lattice_worker.rs,qwen/vision.rs,qwen/worker.rs}`,
//! `src/serving_worker.rs`, and `src/bin/`;
//! these trees stay disjoint from the origin directories `model/` and
//! `forward/metal_*`. A same-file local item therefore cannot hide an origin
//! crossing. Bare-name lookup remains last, after full path resolution.
//!
//! ## Fifth narrowing: a file mounted from more than one module tree
//!
//! Rust's `#[path]` attribute lets a single physical source file be
//! `mod`-mounted from two independent module trees at once -- e.g. a
//! package-local binary's own `mod name;` pointing at a file the library
//! ALSO mounts under `lib.rs`'s tree. The file-discovery walker records
//! every additional (namespace, module_path) mounting of an
//! already-discovered file in `Walker::extra_mounts` (walked and indexed
//! in its own right, since its own nested items need the new namespace
//! too) rather than silently keeping only the first-discovered mounting --
//! which is what a canonical-path-keyed map would otherwise do. Residual
//! risk: this only detects a second mounting reached through this
//! resolver's own root/frontend walk; a file mounted exclusively from a
//! target this resolver never walks would still be invisible.
//!
//! ## Sixth narrowing: macro-body coverage is token-shaped, not syntactic
//!
//! An unclassifiable macro invocation's body is not syn-parseable as items
//! or expressions (that is what makes it unclassifiable), so the only
//! coverage this resolver has for a reference written inside one is a
//! scan of the macro's own raw token stream for `Ident (:: Ident)+` runs,
//! each fed through the same `resolve_reference` an ordinary parsed path
//! uses (`path_shaped_token_runs`). This is deliberately over-reporting
//! rather than under-reporting, and it is not a parse: it can MISS a real
//! origin-type reference, and this is not a hypothetical residual risk --
//! name it plainly:
//!
//! - A reference built by token pasting (`concat_idents!`-style, or any
//!   macro that assembles a path from separately-tokenized pieces) has no
//!   `Ident (:: Ident)+` run in the raw stream for this scan to find --
//!   the identifier does not exist as such until a LATER macro expansion
//!   constructs it, and this resolver never expands macros.
//! - A reference produced by a NESTED macro expansion (a macro invoked
//!   from inside this macro's own body, whose expansion is what actually
//!   names an origin type) is invisible for the same reason: the name
//!   exists only after expansion, and this scan reads tokens, not
//!   expanded output.
//! - Only MULTI-segment runs (two or more `Ident`s joined by `::`) are
//!   collected. A bare single `Ident` head is deliberately excluded --
//!   inside a macro body it is usually a local binding (a `select!` arm's
//!   own pattern name, a captured variable), and including it would
//!   require the same in-scope-binding matching the parsed-path arm gets
//!   from `syn`'s own scoping, which this text scan does not have. A bare
//!   name that is genuinely an origin-type reference with no further path
//!   segment is therefore also missed.
//!
//! None of these three gaps fails loudly: a missed reference here simply
//! never becomes a `Crossing` and never becomes an `unresolved` entry --
//! it is silent by construction, the same way an unscanned file would be.

// This test target uses only a subset of the shared support module's exports
// (the rest serve `metal_measurement_lock_contract.rs`'s wider contract);
// `allow(dead_code)` on the module item, not on `source_graph.rs` itself,
// suppresses the resulting per-binary dead-code warnings without touching
// the shared file.
#[path = "support/source_graph.rs"]
#[allow(dead_code)]
mod source_graph;
use source_graph::*;

#[path = "support/pipeline_boundary_packages.rs"]
mod pipeline_boundary_packages;
use pipeline_boundary_packages::*;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::visit::Visit;
use syn::{Expr, Item, Type, UseTree};

/// Sentinel namespace an alias/re-export/fn-return entry can point at to
/// mean "resolves to something external" (currently: a `std`/`core`/
/// `alloc` item reached through a module-local alias). Never a real
/// namespace name (those are always `"crate"`, `"bin::<name>"`, or
/// `"self"`), so it cannot collide. `Resolver::lookup` checks for it
/// before doing an index lookup.
const EXTERNAL_NAMESPACE: &str = "$external";

/// Rust prelude type/trait names: always in scope with no `use`, so they
/// never gain an alias-table entry the way an imported name does. A bare
/// multi-segment reference through one (`String::new()`, `Vec::new()`)
/// falls all the way to the "bare reference to a local item" lookup with
/// nothing else to catch it; this is the fallback for exactly that case,
/// tried only AFTER the local-item lookup has already failed, so a crate
/// item that happens to shadow a prelude name still wins. Not exhaustive
/// (the 2021+ prelude is larger); this list is a bounded approximation
/// covering the names most likely to appear as `Name::method(..)` --
/// residual risk, same class as the origin-directory rule.
fn is_prelude_type(name: &str) -> bool {
    matches!(
        name,
        "String"
            | "Vec"
            | "Box"
            | "Option"
            | "Result"
            | "Default"
            | "Clone"
            | "Iterator"
            | "IntoIterator"
            | "From"
            | "Into"
            | "TryFrom"
            | "TryInto"
            | "AsRef"
            | "AsMut"
            | "ToOwned"
            | "ToString"
    )
}

/// Primitive type names -- part of the language, not an import of any
/// kind, and explicitly named external by the resolution algorithm
/// (module doc comment: "a first segment that is std/core/alloc, a
/// declared extern crate, a primitive, or a local binding is external").
/// Same fallback timing as `is_prelude_type`: tried only after the local-
/// item lookup has failed.
fn is_primitive_type(name: &str) -> bool {
    matches!(
        name,
        "bool"
            | "char"
            | "str"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "f32"
            | "f64"
    )
}

// ---------------------------------------------------------------------
// Report shape
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Crossing {
    relative_path: String,
    resolved_path: String,
    kind: String,
}

#[allow(dead_code)] // `targets` is production-report-only; fixtures leave it empty.
struct BoundaryReport {
    targets: Vec<CargoTargetRoot>,
    modules_read: Vec<PathBuf>,
    crossings: BTreeSet<Crossing>,
    unresolved: Vec<String>,
}

// ---------------------------------------------------------------------
// Namespace-aware module walk (private to this file; does not touch
// source_graph.rs, which stays exactly as shared with the Metal lock
// guard). Mirrors ModuleGraph's file-discovery logic but additionally
// tracks each file's own module-path segments, which module_source_closure
// does not expose and which this resolver needs for self::/super::
// resolution and for keying the item index.
// ---------------------------------------------------------------------

#[derive(Clone)]
struct WalkedFile {
    namespace: String,
    module_path: Vec<String>,
    cfg: CfgFormula,
    file: syn::File,
}

struct Walker<'a> {
    manifest_dir: &'a Path,
    manifest_canonical: PathBuf,
    test_cfg: bool,
    files: BTreeMap<PathBuf, WalkedFile>,
    visiting: BTreeSet<PathBuf>,
    /// Every additional (namespace, module_path) mounting of a physical
    /// file already recorded in `files` under a DIFFERENT mounting --
    /// see `load`'s doc comment on why a single canonical-path-keyed map
    /// cannot record more than one.
    extra_mounts: Vec<(PathBuf, WalkedFile)>,
    /// De-duplicates `extra_mounts` entries: (canonical path, namespace,
    /// module_path) triples already recorded there.
    extra_mount_keys: BTreeSet<(PathBuf, String, Vec<String>)>,
}

impl<'a> Walker<'a> {
    fn new(manifest_dir: &'a Path, test_cfg: bool) -> Result<Self, String> {
        let manifest_canonical = std::fs::canonicalize(manifest_dir).map_err(|reason| {
            format!(
                "could not resolve crate boundary {}: {reason}",
                manifest_dir.display()
            )
        })?;
        Ok(Self {
            manifest_dir,
            manifest_canonical,
            test_cfg,
            files: BTreeMap::new(),
            visiting: BTreeSet::new(),
            extra_mounts: Vec::new(),
            extra_mount_keys: BTreeSet::new(),
        })
    }

    /// Walks `root` into `namespace`, starting at `root_module_path`.
    /// Returns the canonical root path (the caller needs it to compute
    /// frontend membership).
    fn walk_root(
        &mut self,
        root: &Path,
        namespace: &str,
        root_module_path: Vec<String>,
    ) -> Result<PathBuf, String> {
        let canonical = std::fs::canonicalize(root)
            .map_err(|reason| format!("could not resolve root {}: {reason}", root.display()))?;
        let module_dir = root
            .parent()
            .ok_or_else(|| format!("root {} has no parent", root.display()))?;
        self.load(
            &canonical,
            (module_dir, module_dir),
            namespace,
            root_module_path,
            CfgFormula::True,
        )?;
        Ok(canonical)
    }

    fn load(
        &mut self,
        canonical: &Path,
        module_dirs: (&Path, &Path),
        namespace: &str,
        module_path: Vec<String>,
        inherited_cfg: CfgFormula,
    ) -> Result<(), String> {
        if !canonical.starts_with(&self.manifest_canonical) {
            return Err(format!(
                "unclassifiable compiler-selected module outside crate boundary: {}",
                canonical.display()
            ));
        }
        if !self.visiting.insert(canonical.to_path_buf()) {
            return Err(format!(
                "module graph cycle while classifying {}",
                canonical.display()
            ));
        }
        let source = std::fs::read_to_string(canonical).map_err(|reason| {
            format!(
                "could not read compiler-selected module {}: {reason}",
                canonical.display()
            )
        })?;
        let context = canonical
            .strip_prefix(self.manifest_dir)
            .unwrap_or(canonical)
            .to_string_lossy()
            .into_owned();
        let syntax = syn::parse_file(&source).map_err(|reason| {
            format!("{context}: Rust syntax could not be classified: {reason}")
        })?;
        let file_cfg = CfgFormula::all([
            inherited_cfg,
            syn_attributes_formula(&syntax.attrs, self.test_cfg, &context)?,
        ]);
        if !file_cfg.satisfiable()? {
            self.visiting.remove(canonical);
            return Ok(());
        }
        // A physical file reached the SECOND time under a DIFFERENT
        // (namespace, module_path) is not a re-discovery of the same
        // module: Rust's `#[path]` attribute lets one source file be
        // `mod`-mounted from two entirely separate module trees at once
        // (a package-local binary's own `mod name;` pointing at a file
        // the library ALSO mounts under `lib.rs`'s tree is the concrete
        // case that surfaced this -- `weights/f16_encode.rs`, mounted
        // both as `crate::weights::f16_encode` and, via
        // `src/bin/quantize_q4.rs`'s own `#[path]`-redirected `mod
        // f16_encode;`, as `bin::quantize_q4::f16_encode`). `files` is
        // keyed on canonical path alone, so it can hold only the FIRST
        // mounting; every later, differently-scoped one is recorded in
        // `extra_mounts` instead and walked in its own right (its
        // children need the new namespace/module_path too), so indexing
        // sees every mounting a reference could actually resolve
        // through, not just whichever one the walk reached first.
        if let Some(existing) = self.files.get_mut(canonical) {
            existing.cfg = CfgFormula::any([existing.cfg.clone(), file_cfg.clone()]);
            let is_new_mounting =
                existing.namespace != namespace || existing.module_path != module_path;
            self.visiting.remove(canonical);
            if is_new_mounting
                && self.extra_mount_keys.insert((
                    canonical.to_path_buf(),
                    namespace.to_string(),
                    module_path.clone(),
                ))
            {
                self.extra_mounts.push((
                    canonical.to_path_buf(),
                    WalkedFile {
                        namespace: namespace.to_string(),
                        module_path: module_path.clone(),
                        cfg: file_cfg.clone(),
                        file: syntax.clone(),
                    },
                ));
                self.walk_items(
                    &syntax.items,
                    canonical,
                    module_dirs,
                    namespace,
                    module_path,
                    file_cfg,
                )?;
            }
            return Ok(());
        }
        self.files.insert(
            canonical.to_path_buf(),
            WalkedFile {
                namespace: namespace.to_string(),
                module_path: module_path.clone(),
                cfg: file_cfg.clone(),
                file: syntax.clone(),
            },
        );
        self.walk_items(
            &syntax.items,
            canonical,
            module_dirs,
            namespace,
            module_path,
            file_cfg,
        )?;
        self.visiting.remove(canonical);
        Ok(())
    }

    fn walk_items(
        &mut self,
        items: &[Item],
        source_path: &Path,
        module_dirs: (&Path, &Path),
        namespace: &str,
        module_path: Vec<String>,
        inherited_cfg: CfgFormula,
    ) -> Result<(), String> {
        let (module_dir, path_attribute_dir) = module_dirs;
        let context = source_path
            .strip_prefix(self.manifest_dir)
            .unwrap_or(source_path)
            .to_string_lossy()
            .into_owned();
        for item in items {
            let Item::Mod(module) = item else { continue };
            let module_cfg = CfgFormula::all([
                inherited_cfg.clone(),
                syn_attributes_formula(&module.attrs, self.test_cfg, &context)?,
            ]);
            if !module_cfg.satisfiable()? {
                continue;
            }
            let mut child_module_path = module_path.clone();
            child_module_path.push(module.ident.to_string());
            if let Some((_, contents)) = &module.content {
                if literal_module_path(&module.attrs, &context)?.is_some() {
                    return Err(format!(
                        "{context}: unclassifiable path attribute on inline module `{}`",
                        module.ident
                    ));
                }
                let inline_dir = module_dir.join(module.ident.to_string());
                self.walk_items(
                    contents,
                    source_path,
                    (&inline_dir, &inline_dir),
                    namespace,
                    child_module_path,
                    module_cfg,
                )?;
                continue;
            }
            let direct_path = literal_module_path(&module.attrs, &context)?;
            let target = if let Some(relative) = &direct_path {
                path_attribute_dir.join(relative)
            } else {
                let flat = module_dir.join(format!("{}.rs", module.ident));
                let nested = module_dir.join(module.ident.to_string()).join("mod.rs");
                match (flat.exists(), nested.exists()) {
                    (true, false) => flat,
                    (false, true) => nested,
                    (true, true) => {
                        return Err(format!(
                            "{context}: ambiguous compiler-selected module `{}`",
                            module.ident
                        ));
                    }
                    (false, false) => {
                        return Err(format!(
                            "{context}: compiler-selected module `{}` could not be resolved",
                            module.ident
                        ));
                    }
                }
            };
            let target_canonical = std::fs::canonicalize(&target).map_err(|reason| {
                format!(
                    "unclassifiable compiler-selected module {}: {reason}",
                    target.display()
                )
            })?;
            let child_module_dir = if direct_path.is_some()
                || target.file_name().is_some_and(|name| name == "mod.rs")
            {
                target.parent().unwrap_or(module_dir).to_path_buf()
            } else {
                target
                    .parent()
                    .unwrap_or(module_dir)
                    .join(module.ident.to_string())
            };
            self.load(
                &target_canonical,
                (&child_module_dir, target.parent().unwrap_or(module_dir)),
                namespace,
                child_module_path,
                module_cfg,
            )?;
        }
        Ok(())
    }
}

/// A root's own starting module path: empty if it is itself a known Cargo
/// target (a crate root), else inferred from its own file name.
fn inferred_module_path(
    root_canonical: &Path,
    known_target_paths: &BTreeSet<PathBuf>,
) -> Vec<String> {
    if known_target_paths.contains(root_canonical) {
        return Vec::new();
    }
    let stem = root_canonical
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if stem == "lib" {
        return Vec::new();
    }
    if stem == "mod" {
        return root_canonical
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .map(|n| vec![n.to_string()])
            .unwrap_or_default();
    }
    vec![stem.to_string()]
}

// ---------------------------------------------------------------------
// External-name detection: read (never write) crate_dir/Cargo.toml.
// Absent entirely for fixtures, which have no Cargo.toml.
// ---------------------------------------------------------------------

struct CrateManifestInfo {
    package_name: Option<String>, // underscore form, e.g. "lattice_inference"
    dependency_names: BTreeSet<String>,
}

fn read_crate_manifest_info(crate_dir: &Path) -> CrateManifestInfo {
    let manifest_path = crate_dir.join("Cargo.toml");
    let Ok(text) = std::fs::read_to_string(&manifest_path) else {
        return CrateManifestInfo {
            package_name: None,
            dependency_names: BTreeSet::new(),
        };
    };
    let mut package_name = None;
    let mut dependency_names = BTreeSet::new();
    let mut section = String::new();
    let mut in_package = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            section = trimmed.trim_matches(['[', ']']).to_string();
            in_package = section == "package";
            continue;
        }
        let is_dep_section = section == "dependencies"
            || section == "dev-dependencies"
            || section == "build-dependencies"
            || section.ends_with(".dependencies");
        if in_package
            && package_name.is_none()
            && let Some(rest) = trimmed.strip_prefix("name")
        {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let value = rest.trim().trim_matches('"').trim_matches('\'');
                if !value.is_empty() {
                    package_name = Some(value.replace('-', "_"));
                }
            }
        }
        if is_dep_section && let Some(eq_pos) = trimmed.find('=') {
            let full_key = trimmed[..eq_pos].trim();
            // TOML dotted-key form (`serde_json.workspace = true`) names
            // the crate as the FIRST segment; `serde_json = { workspace
            // = true }` names it as the whole key. Splitting on '.' and
            // taking the first segment handles both: for the
            // undotted form `key` already has no '.', so this is a
            // no-op. Without this split, every workspace-inherited
            // dependency written in dotted form (measured: this
            // manifest uses it for `serde`, `serde_json`, `base64`,
            // `sha2`, `tracing`, `half`) was invisible to
            // `dependency_names`, so a fully-qualified reference to any
            // of those crates (`serde_json::to_string`, ...) fell
            // through the "declared extern crate" check and was
            // wrongly reported as unresolved.
            let key = full_key.split('.').next().unwrap_or(full_key).trim();
            let valid_key = !key.is_empty()
                && key
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
            if valid_key {
                dependency_names.insert(key.replace('-', "_"));
            }
        }
    }
    CrateManifestInfo {
        package_name,
        dependency_names,
    }
}

// ---------------------------------------------------------------------
// Item index
// ---------------------------------------------------------------------

/// A resolved re-export/alias/glob target: an optional namespace override
/// (`None` means "same namespace as the reference site") plus the target's
/// absolute segments.
type ResolvedTarget = (Option<String>, Vec<String>);

#[derive(Default)]
struct NamespaceIndex {
    definitions: BTreeMap<String, PathBuf>,
    // absolute introduced path -> (target namespace override, target absolute segments)
    reexports: BTreeMap<String, (Option<String>, Vec<String>)>,
    // (module_path joined "::", local name) -> (target namespace override, target absolute segments)
    aliases: BTreeMap<(String, String), (Option<String>, Vec<String>)>,
    // absolute fn path -> (target namespace override, target absolute segments) of a direct-path return
    fn_returns: BTreeMap<String, (Option<String>, Vec<String>)>,
    // every module path known to exist in this namespace (file-based AND
    // inline), used to tell "names a whole module, no further segments"
    // apart from "names nothing" -- a module is never itself a concrete
    // origin item, so a reference resolving down to exactly one of these
    // is external rather than unresolved.
    module_paths: BTreeSet<Vec<String>>,
    // per DECLARING module path, every glob import's own already-resolved
    // target (namespace override, absolute module path) -- `use super::*;`
    // and `use crate::path::*;` only (an external-crate-prefixed glob is a
    // rarer pattern and stays a documented conservative gap, matching
    // `flatten_use_tree`'s glob-arm comment). Consulted by
    // `Resolver::resolve_reference` only as a last resort, after every
    // higher-priority classification (named alias, local definition) has
    // already failed -- see that function's own comment for why the
    // ordering matters.
    glob_imports: BTreeMap<Vec<String>, Vec<ResolvedTarget>>,
}

fn join_path(module_path: &[String], name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        format!("{}::{}", module_path.join("::"), name)
    }
}

/// Extracts the absolute segments of a direct `Type::Path` return type,
/// with no leading qself and no generic arguments on the terminal segment
/// (anything else is out of scope for the bounded inferred-constructor
/// rule and simply yields None, which the caller treats as "no inference
/// available" rather than a failure).
fn direct_return_segments(ret: &syn::ReturnType) -> Option<Vec<String>> {
    let syn::ReturnType::Type(_, ty) = ret else {
        return None;
    };
    let Type::Path(type_path) = ty.as_ref() else {
        return None;
    };
    if type_path.qself.is_some() {
        return None;
    }
    Some(
        type_path
            .path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect(),
    )
}

fn flatten_use_tree(
    tree: &UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, String)>,
    globs: &mut Vec<Vec<String>>,
) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            flatten_use_tree(&path.tree, prefix, out, globs);
            prefix.pop();
        }
        UseTree::Name(name) => {
            let ident = name.ident.to_string();
            if ident == "self" {
                if let Some(local) = prefix.last().cloned() {
                    out.push((prefix.clone(), local));
                }
            } else {
                let mut segments = prefix.clone();
                segments.push(ident.clone());
                out.push((segments, ident));
            }
        }
        UseTree::Rename(rename) => {
            let mut segments = prefix.clone();
            segments.push(rename.ident.to_string());
            out.push((segments, rename.rename.to_string()));
        }
        UseTree::Glob(_) => {
            // No named alias comes out of a glob -- what a glob DOES carry
            // is its own prefix (e.g. `super`, or `crate::foo::bar`),
            // recorded as-written here. Resolving that prefix into an
            // absolute module path happens at the call site, which knows
            // the declaring module and can reuse `canonicalize_use_target`
            // the same way a named leaf's first segment is resolved; this
            // function only flattens syntax, it does not resolve anything.
            // See `NamespaceIndex::glob_imports` and
            // `Resolver::resolve_reference`'s glob-fallback comment for
            // where the recorded prefix is actually used, and for the
            // residual gap (an external-crate-prefixed glob is not
            // resolved here at all, since that needs the same uniform-
            // paths classification a NAMED leaf gets, which has no final
            // item name to classify against for a bare glob).
            globs.push(prefix.clone());
        }
        UseTree::Group(group) => {
            for item in &group.items {
                flatten_use_tree(item, prefix, out, globs);
            }
        }
    }
}

/// Collects every `ItemUse`, at any nesting depth (module level, or
/// scoped inside a function/block body), via a full recursive syn visit --
/// not the flat top-level-items loop `build_index_with_paths` otherwise
/// uses for definitions.
/// Collects every `use` item in a file -- module level, function-body-local,
/// and nested arbitrarily deep inside inline `mod name { ... }` blocks --
/// each attributed to the path of its INNERMOST enclosing module rather
/// than uniformly to the file's own top-level path. A `use` written inside
/// `mod grammar { use foo::Bar; ... }` belongs to `grammar`'s own alias
/// table, not the file's: without this, an inline module's imports would be
/// indexed as if declared one level higher, silently missing the alias
/// table `resolve_reference` actually looks up when scanning a reference
/// written inside that same inline module (whose `home_module_path` is
/// threaded the same way -- see `CrossingScanner::visit_item_mod`).
struct UseCollector<'ast> {
    leaves: Vec<(Vec<String>, &'ast syn::ItemUse)>,
    module_path: Vec<String>,
}

impl<'ast> Visit<'ast> for UseCollector<'ast> {
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        self.leaves.push((self.module_path.clone(), node));
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        // Only an INLINE module's own content deepens the path -- a
        // file-based `mod foo;` has no content here to descend into at
        // all (its items live in a separate file, walked and indexed
        // independently as their own `WalkedFile`), so the default
        // recursion below is already a no-op for it.
        let is_inline = node.content.is_some();
        if is_inline {
            self.module_path.push(node.ident.to_string());
        }
        syn::visit::visit_item_mod(self, node);
        if is_inline {
            self.module_path.pop();
        }
    }
}

fn collect_use_items_with_module_path<'ast>(
    file: &'ast syn::File,
    home_module_path: &[String],
) -> Vec<(Vec<String>, &'ast syn::ItemUse)> {
    let mut collector = UseCollector {
        leaves: Vec::new(),
        module_path: home_module_path.to_vec(),
    };
    collector.visit_file(file);
    collector.leaves
}

/// Flattens `items` into `(effective_module_path, item)` pairs, descending
/// into inline `mod name { ... }` blocks so that a struct/enum/fn/const/etc.
/// declared physically inside one is indexed at its OWN module path rather
/// than being invisible to a flat top-level-items scan. Does not descend
/// into a file-based `mod name;` (no content here to descend into -- its
/// items are indexed independently as their own `WalkedFile`). Mirrors the
/// Walker's own inline-module recursion in `walk_items`, which already does
/// this for module-GRAPH discovery (finding further file-based `mod`
/// declarations nested inside); this is indexing's twin for definitions.
fn flatten_module_items<'ast>(
    items: &'ast [Item],
    module_path: &[String],
    out: &mut Vec<(Vec<String>, &'ast Item)>,
) {
    for item in items {
        if let Item::Mod(module) = item
            && let Some((_, contents)) = &module.content
        {
            let mut child_path = module_path.to_vec();
            child_path.push(module.ident.to_string());
            flatten_module_items(contents, &child_path, out);
            continue;
        }
        out.push((module_path.to_vec(), item));
    }
}

/// Classifies the first segment of a `use`-tree target per Rust 2018+
/// import rules (see the module doc comment): `crate`/`self`/`super`
/// resolve relative to `home_module_path`; a name matching a known
/// namespace (the crate's own package name) switches namespace; anything
/// else is an extern crate reference, known or not, and yields `None`
/// (skip — do not create an alias/reexport entry for it).
fn canonicalize_use_target(
    home_module_path: &[String],
    raw: &[String],
) -> Option<(Option<String>, Vec<String>)> {
    let (first, rest) = raw.split_first()?;
    match first.as_str() {
        "crate" => Some((None, rest.to_vec())),
        "self" => {
            let mut segments = home_module_path.to_vec();
            segments.extend_from_slice(rest);
            Some((None, segments))
        }
        "super" => {
            let mut idx = 0usize;
            while idx < raw.len() && raw[idx] == "super" {
                idx += 1;
            }
            if idx > home_module_path.len() {
                return None;
            }
            let mut segments = home_module_path[..home_module_path.len() - idx].to_vec();
            segments.extend_from_slice(&raw[idx..]);
            Some((None, segments))
        }
        _ => None, // handled by the caller, which knows the namespace map
    }
}

struct CrossingCrateContext {
    crate_name_to_namespace: BTreeMap<String, String>,
    external_crate_names: BTreeSet<String>,
}

/// Finalizes a `use`-tree leaf's target into a canonical (namespace,
/// absolute segments) address, or `None` if it's external (no alias/
/// re-export entry should be created for it).
///
/// This MUST run after every namespace's `definitions` and module-path set
/// are fully known (a second pass over the whole discovered file set),
/// because Rust 2018+ "uniform paths" means an unqualified `use` first
/// segment resolves to EITHER an extern crate OR a sibling item/module
/// declared in the current crate -- and telling those apart requires
/// knowing what the crate actually declares, not just syntax. A `use`
/// finalized against a partially-built index would wrongly treat a
/// same-crate sibling module as an unrecognized extern crate whenever that
/// sibling's own file happened to be discovered later in iteration order.
fn finalize_use_leaf(
    ctx: &CrossingCrateContext,
    indices: &BTreeMap<String, NamespaceIndex>,
    home_namespace: &str,
    home_module_path: &[String],
    raw: &[String],
) -> Option<(Option<String>, Vec<String>)> {
    if let Some(result) = canonicalize_use_target(home_module_path, raw) {
        return Some(result);
    }
    let first = raw.first()?;
    if let Some(ns) = ctx.crate_name_to_namespace.get(first) {
        return Some((Some(ns.clone()), raw[1..].to_vec()));
    }
    // An already-finalized alias in the SAME module (this fires both for
    // a genuinely chained `use` -- rare -- and, more commonly, for a
    // fn-return-type segment list that is itself just a bare name already
    // brought into scope by an earlier `use` in the same file, which is
    // exactly what a return type written as `-> ConcreteModel` after
    // `use crate::model::ConcreteModel;` looks like to this function).
    if let Some((ns, target)) = indices
        .get(home_namespace)
        .and_then(|index| {
            index
                .aliases
                .get(&(home_module_path.join("::"), first.clone()))
        })
        .cloned()
    {
        let mut combined = target;
        combined.extend(raw[1..].iter().cloned());
        return Some((ns, combined));
    }
    // Rust 2018+ uniform paths: an unqualified `use` first segment can
    // also name a sibling module or item declared directly in the current
    // module. Check both before falling back to "unrecognized extern
    // crate, treat as external".
    let mut local = home_module_path.to_vec();
    local.push(first.clone());
    let names_a_sibling_module = indices.get(home_namespace).is_some_and(|index| {
        index
            .module_paths
            .iter()
            .any(|path| path.starts_with(&local))
    });
    let names_a_sibling_item = indices
        .get(home_namespace)
        .is_some_and(|index| index.definitions.contains_key(&local.join("::")));
    if names_a_sibling_module || names_a_sibling_item {
        let mut segments = home_module_path.to_vec();
        segments.extend(raw.iter().cloned());
        return Some((None, segments));
    }
    // Anything else -- `std`/`core`/`alloc`, a declared dependency, or an
    // unrecognized name (a dev/optional dependency this resolver's
    // Cargo.toml scan did not enumerate, most likely) -- is an extern
    // crate reference. Recorded as an alias pointing at the `$external`
    // sentinel namespace (which `Resolver::lookup` short-circuits to
    // `External`) rather than skipped outright: skipping it would leave a
    // LATER bare multi-segment reference through the imported name (`use
    // tokio::sync::mpsc; ... mpsc::unbounded_channel()`, `use std::path::
    // PathBuf; ... PathBuf::from(x)`) with no alias to resolve through --
    // such a reference never reaches the single-segment "unknown -> local
    // binding" fallback either, since it has more than one segment, so it
    // fell through to "bare reference to something in the current module"
    // and was wrongly reported as unresolved. Measured against the real
    // crate: every import-then-multi-segment-referenced external item
    // (`PathBuf::from`, `Instant::now`, `mpsc::unbounded_channel`,
    // `base64::Engine`, ...) produced exactly this false unresolved before
    // this fix existed.
    let _ = ctx.external_crate_names.contains(first);
    Some((Some(EXTERNAL_NAMESPACE.to_string()), Vec::new()))
}

// ---------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------

enum Resolved {
    External,
    /// The defining file, plus the fully-resolved (namespace, absolute
    /// segments) address actually found -- which is what makes the
    /// reported crossing carry the RESOLVED item path rather than the
    /// as-written reference text (an alias, a re-export hop, or a bare
    /// same-module name all read differently at the use site than at the
    /// definition).
    Definition(PathBuf, String, Vec<String>),
    Unresolved,
}

struct Resolver<'a> {
    ctx: &'a CrossingCrateContext,
    indices: &'a BTreeMap<String, NamespaceIndex>,
}

impl<'a> Resolver<'a> {
    /// Entry point: resolves a raw, as-written path found while scanning a
    /// frontend file (a general Path/Expr/Type reference, NOT a `use`
    /// tree -- those go through `finalize_use_leaf` at index-build time).
    fn resolve_reference(
        &self,
        home_namespace: &str,
        home_module_path: &[String],
        raw: &[String],
        depth_budget: u32,
    ) -> Resolved {
        if raw.is_empty() {
            return Resolved::External;
        }
        if depth_budget == 0 {
            return Resolved::Unresolved;
        }
        let first = raw[0].as_str();
        if raw.len() == 1 && first == "self" {
            // A BARE, single-segment `self` in reference position is the
            // current-binding VALUE (e.g. a builder method returning
            // `self`), not a module path -- unlike `self::x` (len > 1),
            // which is genuine use-tree/path module addressing and is
            // handled correctly below via `canonicalize_use_target`.
            // Without this, `canonicalize_use_target` reads a bare `self`
            // as "the current module, zero further segments", which
            // `Resolver::lookup` then rejects as Unresolved on an empty
            // segment list -- a local binding wrongly reported as a
            // completeness failure.
            return Resolved::External;
        }
        if matches!(first, "std" | "core" | "alloc") {
            return Resolved::External;
        }
        if let Some((ns, rest)) = canonicalize_use_target(home_module_path, raw) {
            let ns = ns.unwrap_or_else(|| home_namespace.to_string());
            return self.lookup(&ns, &rest, depth_budget);
        }
        if first == "super" {
            // canonicalize_use_target returned None: too many leading
            // super:: segments to resolve against home_module_path.
            return Resolved::Unresolved;
        }
        if let Some(ns) = self.ctx.crate_name_to_namespace.get(first) {
            return self.lookup(ns, &raw[1..], depth_budget);
        }
        if raw.len() == 1 {
            // Single-segment identifier not otherwise classified: a local
            // binding (parameter, let-variable, generic parameter, `Self`
            // with no concrete self type on record, or a prelude/
            // primitive name). See module doc comment for why this is
            // sound: a local binding can never be the first segment of a
            // longer path in valid Rust, so this cannot mask a multi-
            // segment crossing.
            //
            // Still worth a direct index probe first, so a bare reference
            // to a same-module item (no `use` needed for that) resolves
            // to its real definition rather than being written off.
            let home_key = join_path(home_module_path, first);
            if let Some(namespace_index) = self.indices.get(home_namespace) {
                if let Some(file) = namespace_index.definitions.get(&home_key) {
                    let mut segments = home_module_path.to_vec();
                    segments.push(first.to_string());
                    return Resolved::Definition(
                        file.clone(),
                        home_namespace.to_string(),
                        segments,
                    );
                }
                if let Some((ns, rest)) = namespace_index
                    .aliases
                    .get(&(home_module_path.join("::"), first.to_string()))
                    .cloned()
                {
                    let ns = ns.unwrap_or_else(|| home_namespace.to_string());
                    return self.lookup(&ns, &rest, depth_budget - 1);
                }
            }
            return Resolved::External;
        }
        if self.ctx.external_crate_names.contains(first)
            && !self.ctx.crate_name_to_namespace.contains_key(first)
        {
            return Resolved::External;
        }
        // Multi-segment, not crate/self/super/known-namespace/declared
        // extern, and not resolved via a same-module alias/definition
        // lookup below: check the alias table first (module-local `use`
        // aliasing), then fall through.
        if let Some(namespace_index) = self.indices.get(home_namespace)
            && let Some((ns, target)) = namespace_index
                .aliases
                .get(&(home_module_path.join("::"), first.to_string()))
                .cloned()
        {
            let ns = ns.unwrap_or_else(|| home_namespace.to_string());
            let mut combined = target;
            combined.extend(raw[1..].iter().cloned());
            return self.lookup(&ns, &combined, depth_budget - 1);
        }
        // Bare reference to something defined directly in the current
        // module (no `use` needed for that in Rust) -- tried FIRST, so a
        // local item that happens to share a name with a prelude type, or
        // with a name a glob import would also bring into scope, wins
        // over both fallbacks below (matching Rust's own shadowing rule:
        // a local item or a specific `use` always beats a glob import).
        let mut local = home_module_path.to_vec();
        local.extend(raw.iter().cloned());
        let local_result = self.lookup(home_namespace, &local, depth_budget);
        if !matches!(local_result, Resolved::Unresolved) {
            return local_result;
        }
        // A name brought into scope only by `use super::*;` (or `use
        // crate::path::*;`) carries no per-name alias entry --
        // `flatten_use_tree` deliberately does not enumerate a glob's
        // members, since doing so would need a second index pass of its
        // own target, the same one `finalize_use_leaf` needs for a NAMED
        // leaf's unqualified first segment. What IS known is the glob's
        // own already-resolved target module, recorded per DECLARING
        // module at index-build time (`NamespaceIndex::glob_imports`):
        // tried only after the direct local lookup above has already
        // failed -- matching Rust's own priority order, where a local
        // item or a specific `use` always beats a glob import -- so this
        // can only turn a previously-Unresolved entry into whatever
        // another module's own item actually is, never mask a genuine
        // crossing the local lookup would have found.
        //
        // This exists because correctly scoping an inline module's OWN
        // `home_module_path` (rather than flattening it to the file's
        // top-level path) means a reference inside `mod tests { use
        // super::*; ... }` no longer coincidentally resolves through
        // whatever alias happens to be registered one level up -- the
        // inline-module fix makes the scoping correct, and this makes the
        // now-exposed glob import resolvable again instead of a false
        // completeness failure. See `resolve_via_glob`'s own doc comment
        // for why one hop is not enough.
        let glob_result =
            self.resolve_via_glob(home_namespace, home_module_path, raw, depth_budget);
        if !matches!(glob_result, Resolved::Unresolved) {
            return glob_result;
        }
        if is_prelude_type(first) || is_primitive_type(first) {
            return Resolved::External;
        }
        local_result
    }

    /// Chases a name through the glob imports (`use super::*;`, `use
    /// crate::path::*;`) declared AT `module_path` in `namespace`, trying
    /// each glob's own already-resolved target module as if `raw` had
    /// been written fully qualified through it.
    ///
    /// A single hop is not enough: a test helper module nested INSIDE a
    /// `mod tests { use super::*; ... }` block (`mod some_test { use
    /// super::*; ... }`) has its own glob pointing at `tests`, which
    /// re-exposes nothing of its own -- the name is only actually defined
    /// two hops up, at the module `tests` itself glob-imports from. So a
    /// hop that fails is retried through THAT target's own glob imports,
    /// recursively, bounded by the same `depth_budget` every other
    /// chase in this resolver uses to guarantee termination on a cycle.
    fn resolve_via_glob(
        &self,
        namespace: &str,
        module_path: &[String],
        raw: &[String],
        depth_budget: u32,
    ) -> Resolved {
        if depth_budget == 0 {
            return Resolved::Unresolved;
        }
        let Some(namespace_index) = self.indices.get(namespace) else {
            return Resolved::Unresolved;
        };
        let Some(globs) = namespace_index.glob_imports.get(module_path) else {
            return Resolved::Unresolved;
        };
        for (target_ns, target_path) in globs {
            let ns = target_ns.clone().unwrap_or_else(|| namespace.to_string());
            let mut combined = target_path.clone();
            combined.extend(raw.iter().cloned());
            let direct = self.lookup(&ns, &combined, depth_budget - 1);
            if !matches!(direct, Resolved::Unresolved) {
                return direct;
            }
            let transitive = self.resolve_via_glob(&ns, target_path, raw, depth_budget - 1);
            if !matches!(transitive, Resolved::Unresolved) {
                return transitive;
            }
        }
        Resolved::Unresolved
    }

    /// Looks up an already-namespace-qualified absolute path by
    /// longest-prefix match against definitions, then re-exports
    /// (chasing bounded by `depth_budget`). No further alias
    /// reclassification happens here: by construction, everything reaching
    /// this function is already canonical.
    fn lookup(&self, namespace: &str, segments: &[String], depth_budget: u32) -> Resolved {
        if namespace == EXTERNAL_NAMESPACE {
            return Resolved::External;
        }
        if segments.is_empty() {
            return Resolved::Unresolved;
        }
        let Some(namespace_index) = self.indices.get(namespace) else {
            return Resolved::Unresolved;
        };
        for prefix_len in (1..=segments.len()).rev() {
            let key = segments[..prefix_len].join("::");
            if let Some(file) = namespace_index.definitions.get(&key) {
                return Resolved::Definition(
                    file.clone(),
                    namespace.to_string(),
                    segments[..prefix_len].to_vec(),
                );
            }
            if let Some((target_ns, target_segments)) = namespace_index.reexports.get(&key) {
                if depth_budget == 0 {
                    return Resolved::Unresolved;
                }
                let ns = target_ns.clone().unwrap_or_else(|| namespace.to_string());
                let mut combined = target_segments.clone();
                combined.extend(segments[prefix_len..].iter().cloned());
                return self.lookup(&ns, &combined, depth_budget - 1);
            }
            // The alias table is keyed by (declaring module path, local
            // name), never by absolute item path -- an alias's declaring
            // module is everything in this prefix EXCEPT its own last
            // segment (the aliased name itself). `crate::backend` strips
            // (via `canonicalize_use_target`) to segments `["backend"]` at
            // the crate root, so a `use lattice_inference::model_format as
            // backend;` declared AT that root registers under
            // `("", "backend")` -- without this check, `crate::` strips a
            // reference down to a bare alias name and hands it to `lookup`,
            // which only ever consulted `definitions`/`reexports`, so a
            // `use` that points at ANOTHER module's alias (rather than a
            // real definition or a `pub use` re-export) resolved to
            // nothing every time, whether reached directly
            // (`crate::backend::ModelFormat`) or through a second,
            // sibling-module `use crate::backend;` that re-imports it.
            let declaring_path = segments[..prefix_len - 1].join("::");
            let aliased_name = &segments[prefix_len - 1];
            if let Some((target_ns, target_segments)) = namespace_index
                .aliases
                .get(&(declaring_path, aliased_name.clone()))
            {
                if depth_budget == 0 {
                    return Resolved::Unresolved;
                }
                let ns = target_ns.clone().unwrap_or_else(|| namespace.to_string());
                let mut combined = target_segments.clone();
                combined.extend(segments[prefix_len..].iter().cloned());
                return self.lookup(&ns, &combined, depth_budget - 1);
            }
            // A PUBLIC glob re-export (`pub use self::bert::*;`) puts an
            // item into its declaring module's namespace exactly the way a
            // private `use super::*;` puts a name into a function-body
            // scope -- neither is enumerable by name at index-build time
            // (see `flatten_use_tree`'s glob-arm comment), so neither has a
            // `definitions`/`reexports`/`aliases` entry keyed on the
            // introduced name. `model/mod.rs` has ONLY `pub use
            // self::bert::*;` for `BertModel` -- no named re-export exists
            // anywhere for it -- so `crate::model::BertModel` is
            // unreachable through the three checks above no matter how
            // long this loop runs, and a lookup that never tries the
            // GLOB table at the declaring module (`segments[..prefix_len
            // - 1]`, i.e. "model" here) fails on every real crate that
            // uses this extremely common re-export idiom. Reuses
            // `resolve_via_glob` (already transitive, already
            // depth-bounded) rather than a second copy of the chase.
            let declaring_module = &segments[..prefix_len - 1];
            let rest = &segments[prefix_len - 1..];
            if depth_budget > 0 {
                let via_glob =
                    self.resolve_via_glob(namespace, declaring_module, rest, depth_budget - 1);
                if !matches!(via_glob, Resolved::Unresolved) {
                    return via_glob;
                }
            }
        }
        // The full segment list names a MODULE itself (e.g. a bare `use
        // crate::backend;` that imports a whole aliased module with no
        // further path), not a concrete item inside one. A module can
        // never itself be a concrete model/config/backend-state item --
        // the origin rule is decided by the DEFINING FILE of a concrete
        // item, which a module-only reference never names -- and any real
        // item reached THROUGH that module is scanned independently
        // wherever it is actually referenced with its own trailing
        // segment. Checked last, after every definitions/reexports/alias
        // prefix has already failed, so this can only ever widen a
        // genuine dead end, never mask a multi-segment miss.
        if namespace_index.module_paths.contains(segments) {
            return Resolved::External;
        }
        Resolved::Unresolved
    }
}

// ---------------------------------------------------------------------
// Crossing scanning
// ---------------------------------------------------------------------

fn is_origin_file(file: &Path, origin_dir_abs: &[PathBuf], origin_glob_abs: &[PathBuf]) -> bool {
    origin_dir_abs.iter().any(|dir| file.starts_with(dir))
        || origin_glob_abs.iter().any(|prefix| {
            file.to_string_lossy()
                .starts_with(prefix.to_string_lossy().as_ref())
        })
}

struct CrossingScanner<'a> {
    resolver: &'a Resolver<'a>,
    home_namespace: String,
    home_module_path: Vec<String>,
    relative_path: String,
    origin_dir_abs: &'a [PathBuf],
    origin_glob_abs: &'a [PathBuf],
    crossings: &'a mut BTreeSet<Crossing>,
    unresolved: &'a mut Vec<String>,
    current_self_ty: Vec<Option<Vec<String>>>,
    /// Names bound as a generic TYPE parameter on the innermost item, impl,
    /// or function currently being visited (one set per nesting level,
    /// pushed/popped alongside the declaration that introduces it). A path
    /// whose first segment names one of these is never a concrete model
    /// type -- see the module doc's third narrowing.
    generic_scopes: Vec<BTreeSet<String>>,
    /// The innermost `impl ... { type Name = RHS; }` block's own
    /// associated-type declarations, name -> RHS's raw path segments (only
    /// captured when the RHS is itself a plain `Type::Path` with no
    /// qself -- anything else contributes no substitution and falls
    /// through to the ordinary `Self` self-type substitution below). Two
    /// different impls for the same concrete type can each give `Value` a
    /// different meaning, so this is scoped to the CURRENT impl block via a
    /// stack, never looked up globally by self-type name.
    assoc_type_scopes: Vec<BTreeMap<String, Vec<String>>>,
    /// Every name introduced by an item declaration anywhere in THIS file
    /// (`collect_local_item_names`'s doc comment covers why a bare name,
    /// with no path or scope, is a sound check here). Consulted only as
    /// the LAST resort, after `resolve_reference` has already failed every
    /// other classification -- see the module doc's fourth narrowing.
    local_item_names: &'a BTreeSet<String>,
}

impl<'a> CrossingScanner<'a> {
    fn record(&mut self, raw: &[String], kind: &str) {
        match self
            .resolver
            .resolve_reference(&self.home_namespace, &self.home_module_path, raw, 8)
        {
            Resolved::External => {}
            Resolved::Definition(file, namespace, segments) => {
                if is_origin_file(&file, self.origin_dir_abs, self.origin_glob_abs) {
                    self.crossings.insert(Crossing {
                        relative_path: self.relative_path.clone(),
                        resolved_path: format!("{namespace}::{}", segments.join("::")),
                        kind: kind.to_string(),
                    });
                }
            }
            Resolved::Unresolved => {
                if raw
                    .first()
                    .is_some_and(|first| self.local_item_names.contains(first))
                {
                    return;
                }
                self.unresolved.push(format!(
                    "{}: unresolved reference `{}` ({kind})",
                    self.relative_path,
                    raw.join("::")
                ));
            }
        }
    }

    fn path_segments(path: &syn::Path) -> Vec<String> {
        path.segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect()
    }

    /// The generic TYPE parameter names (never lifetimes, never const
    /// generics -- neither can be the head of a multi-segment reference)
    /// declared directly on `generics`.
    fn generic_type_names(generics: &syn::Generics) -> BTreeSet<String> {
        generics
            .params
            .iter()
            .filter_map(|param| match param {
                syn::GenericParam::Type(type_param) => Some(type_param.ident.to_string()),
                _ => None,
            })
            .collect()
    }

    /// An impl block's own directly-declared associated types (`type Name =
    /// RHS;`), RHS captured only when it is a plain path with no qself.
    fn assoc_type_targets(items: &[syn::ImplItem]) -> BTreeMap<String, Vec<String>> {
        items
            .iter()
            .filter_map(|item| match item {
                syn::ImplItem::Type(assoc) => match &assoc.ty {
                    Type::Path(type_path) if type_path.qself.is_none() => Some((
                        assoc.ident.to_string(),
                        Self::path_segments(&type_path.path),
                    )),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }
}

impl<'ast> Visit<'ast> for CrossingScanner<'_> {
    fn visit_path(&mut self, node: &'ast syn::Path) {
        let mut segments = Self::path_segments(node);
        if segments.first().map(String::as_str) == Some("Self") {
            // `Self::name` inside an impl block first checks whether `name`
            // is an associated type THIS impl declares itself (`type Value
            // = Vec<Message>;`) -- if so, the reference is to that RHS
            // type, not to the concrete self type, and two different impls
            // for the same struct can give the same associated-type name
            // different RHS values, which is why this is a per-impl lookup
            // rather than the self-type substitution below. Only when no
            // such local associated type exists (an inherent method like
            // `Self::new()`, or a name inherited from the trait's own
            // default) does `Self` mean the concrete self type.
            let assoc_replacement = if segments.len() >= 2 {
                self.assoc_type_scopes
                    .iter()
                    .rev()
                    .find_map(|scope| scope.get(&segments[1]))
                    .map(|target| {
                        let mut replaced = target.clone();
                        replaced.extend(segments[2..].iter().cloned());
                        replaced
                    })
            } else {
                None
            };
            if let Some(replaced) = assoc_replacement {
                segments = replaced;
            } else if let Some(Some(self_ty)) = self.current_self_ty.last() {
                let mut replaced = self_ty.clone();
                replaced.extend(segments.into_iter().skip(1));
                segments = replaced;
            }
        }
        // A path headed by an in-scope generic TYPE parameter (`D::Error`,
        // `A::Error::custom`) can never name a concrete model type: the
        // parameter is resolved by whoever calls this function, and a call
        // site that supplies an origin type as the concrete argument would
        // itself name that origin type explicitly, in a path this scan
        // already visits independently. See the module doc's third
        // narrowing.
        if segments.first().is_some_and(|first| {
            self.generic_scopes
                .iter()
                .any(|scope| scope.contains(first))
        }) {
            syn::visit::visit_path(self, node);
            return;
        }
        self.record(&segments, "path");
        syn::visit::visit_path(self, node);
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        let mut leaves = Vec::new();
        // A glob import names no specific item, so it produces nothing to
        // `record()` here -- any actual origin item it brings into scope
        // is caught independently wherever it is later referenced by
        // name, via `Resolver::resolve_reference`'s own glob fallback.
        let mut globs = Vec::new();
        flatten_use_tree(&node.tree, &mut Vec::new(), &mut leaves, &mut globs);
        for (raw, _local) in leaves {
            self.record(&raw, "use");
        }
        // Deliberately do not call the default visitor: a UseTree is not a
        // Path and carries nothing else worth visiting for this scan.
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        // `home_module_path` is threaded one level deeper for an INLINE
        // module's own content, mirroring `UseCollector`'s indexing-side
        // twin: a bare/relative reference written inside `mod grammar {
        // ... }` resolves relative to `grammar`'s own path, not the
        // enclosing file's top-level path. A file-based `mod foo;` has no
        // content here to descend into at all (its items are scanned
        // independently as their own file, with their own home path built
        // from scratch), so this is a no-op for it.
        let is_inline = node.content.is_some();
        if is_inline {
            self.home_module_path.push(node.ident.to_string());
        }
        syn::visit::visit_item_mod(self, node);
        if is_inline {
            self.home_module_path.pop();
        }
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let self_ty = match node.self_ty.as_ref() {
            Type::Path(type_path) if type_path.qself.is_none() => {
                Some(Self::path_segments(&type_path.path))
            }
            _ => None,
        };
        self.current_self_ty.push(self_ty);
        self.generic_scopes
            .push(Self::generic_type_names(&node.generics));
        self.assoc_type_scopes
            .push(Self::assoc_type_targets(&node.items));
        syn::visit::visit_item_impl(self, node);
        self.assoc_type_scopes.pop();
        self.generic_scopes.pop();
        self.current_self_ty.pop();
    }
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.generic_scopes
            .push(Self::generic_type_names(&node.sig.generics));
        syn::visit::visit_item_fn(self, node);
        self.generic_scopes.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.generic_scopes
            .push(Self::generic_type_names(&node.sig.generics));
        syn::visit::visit_impl_item_fn(self, node);
        self.generic_scopes.pop();
    }

    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let Some(init) = &node.init
            && let Expr::Call(call) = init.expr.as_ref()
            && let Expr::Path(func_path) = call.func.as_ref()
            && func_path.qself.is_none()
        {
            let segments = Self::path_segments(&func_path.path);
            self.check_inferred_constructor(&segments);
        }
        syn::visit::visit_local(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        // Sixth narrowing (module doc comment): `node.tokens` is opaque to
        // syn for a macro `unclassifiable_macro` cannot otherwise classify
        // -- it never parses as items or exprs, which is exactly why the
        // classifier below says "unclassifiable" rather than resolving
        // anything itself. The token-shaped fallback scans the RAW token
        // stream for multi-segment `Ident (:: Ident)+` runs and feeds each
        // through the SAME `record` the parsed-path arm uses, so whatever
        // resolves is classified exactly as it would be anywhere else in
        // this file, and a run that itself fails to resolve still hard-
        // fails via `record`'s own `Unresolved` arm -- the generic
        // "unclassifiable macro invocation" message is superseded by this
        // finer-grained, per-reference accounting, not stacked on top of
        // it.
        if unclassifiable_macro(node).is_some() {
            for raw in path_shaped_token_runs(node) {
                self.record(&raw, "macro-token");
            }
        }
        syn::visit::visit_macro(self, node);
    }

    fn visit_item_macro(&mut self, _node: &'ast syn::ItemMacro) {
        // Deliberately do not descend: item-position macro completeness
        // (excluding `macro_rules!` definitions themselves, which are not
        // an invocation) is already checked once per file via
        // `unclassifiable_item_macro` in the caller. The default recursion
        // here would call `visit_macro` on a `macro_rules!` definition's
        // OWN `mac` field, whose `path` is literally `macro_rules` (syn
        // represents a macro_rules! item as an invocation of that name) --
        // `unclassifiable_macro` does not special-case that path, so
        // without this override every file that DEFINES a macro would be
        // wrongly flagged as containing an unclassifiable invocation.
    }

    fn visit_vis_restricted(&mut self, _node: &'ast syn::VisRestricted) {
        // Deliberately do not descend: `pub(crate)`, `pub(super)`, and
        // `pub(in some::path)` all carry a `Path` in `VisRestricted::path`
        // purely as a visibility scope, and syn's default
        // `visit_vis_restricted` recurses into it via `visit_path` like any
        // other path. That path never names or constructs a value -- it is
        // metadata about who may see the item, not a dependency on
        // anything -- so without this override every `pub(crate)` field or
        // function in a scanned file recorded a spurious bare `crate`
        // reference (and `pub(super)` a bare `super`), which cannot resolve
        // to a definition and was wrongly flagged as an unresolved
        // completeness failure. Confirmed: `visit_vis_restricted`'s
        // generated default body is exactly `v.visit_path(&*node.path)`
        // (syn 2.0's `gen/visit.rs`), so overriding this one method is
        // sufficient to stop the funnel without touching `visit_path`
        // itself or any other visibility variant (`pub`/inherited carry no
        // path at all).
    }
}

impl CrossingScanner<'_> {
    fn check_inferred_constructor(&mut self, raw: &[String]) {
        // Bounded to a direct path call to an indexed function whose
        // declared return type is itself resolvable to an origin
        // definition. Resolve the CALLEE first; if it lands on a function
        // definition in some namespace, consult that namespace's
        // fn_returns table for the function's own key.
        let resolved_ns_and_key = self.resolve_callee_key(raw);
        let Some((mut ns, mut key)) = resolved_ns_and_key else {
            return;
        };
        // A callee named through a `pub use` re-export (a facade path to a
        // function another package defines) has no fn_returns row under the
        // re-exporting namespace, so follow the re-export to the definition.
        if !self
            .resolver
            .indices
            .get(&ns)
            .is_some_and(|index| index.fn_returns.contains_key(&key))
        {
            let segments: Vec<String> = key.split("::").map(str::to_string).collect();
            if let Resolved::Definition(_, defining_ns, defining_segments) =
                self.resolver.lookup(&ns, &segments, 8)
            {
                ns = defining_ns;
                key = defining_segments.join("::");
            }
        }
        let Some(namespace_index) = self.resolver.indices.get(&ns) else {
            return;
        };
        let Some((target_ns, target_segments)) = namespace_index.fn_returns.get(&key).cloned()
        else {
            return;
        };
        let target_ns = target_ns.unwrap_or(ns);
        if let Resolved::Definition(file, resolved_ns, resolved_segments) =
            self.resolver.lookup(&target_ns, &target_segments, 8)
            && is_origin_file(&file, self.origin_dir_abs, self.origin_glob_abs)
        {
            self.crossings.insert(Crossing {
                relative_path: self.relative_path.clone(),
                resolved_path: format!("{resolved_ns}::{}", resolved_segments.join("::")),
                kind: "inferred-constructor".to_string(),
            });
        }
    }

    /// Resolves `raw` (a call callee path) down to the (namespace, key)
    /// pair that would index it in `fn_returns`, without emitting a
    /// crossing/unresolved entry for the callee path itself (that already
    /// happens separately via the normal `visit_path` traversal).
    fn resolve_callee_key(&self, raw: &[String]) -> Option<(String, String)> {
        if raw.is_empty() {
            return None;
        }
        if let Some((ns, rest)) = canonicalize_use_target(&self.home_module_path, raw) {
            let ns = ns.unwrap_or_else(|| self.home_namespace.clone());
            return Some((ns, rest.join("::")));
        }
        if let Some(ns) = self
            .resolver
            .ctx
            .crate_name_to_namespace
            .get(raw[0].as_str())
        {
            return Some((ns.clone(), raw[1..].join("::")));
        }
        // Single-segment calls fall through to the alias/local-module
        // checks below like everything else: an aliased or same-module
        // free function is exactly what a bare `f()` call names.
        let namespace_index = self.resolver.indices.get(&self.home_namespace)?;
        if let Some((ns, target)) = namespace_index
            .aliases
            .get(&(self.home_module_path.join("::"), raw[0].clone()))
            .cloned()
        {
            let ns = ns.unwrap_or_else(|| self.home_namespace.clone());
            let mut combined = target;
            combined.extend(raw[1..].iter().cloned());
            return Some((ns, combined.join("::")));
        }
        let mut local = self.home_module_path.clone();
        local.extend(raw.iter().cloned());
        Some((self.home_namespace.clone(), local.join("::")))
    }
}

// ---------------------------------------------------------------------
// analyze_boundary
// ---------------------------------------------------------------------

struct RawUseLeaf {
    namespace: String,
    module_path: Vec<String>,
    raw: Vec<String>,
    local_name: String,
    is_pub: bool,
}

/// Every name introduced by an item declaration ANYWHERE in a scanned
/// file, at ANY nesting depth -- including a block-scoped item declared
/// inside a function body (`fn f() { enum Local { ... } ... }`), which
/// `flatten_module_items` never sees (it only descends into inline `mod
/// name { ... }`, never a function's own block). syn's default `Visit`
/// recursion already reaches a function body's nested items on its own
/// (a block is walked like any other syntax tree, and each `Stmt::Item`
/// inside it is visited), so this collector only needs to record one name
/// per item kind and let the default recursion carry it everywhere --
/// it never needs to track a path, unlike every other index in this file.
///
/// The bare-name fallback's roots (`src/serve`, `src/bin/*`,
/// `src/serving_worker.rs`, and provider files `src/serving_provider/providers/{bert.rs,bert/embedding.rs,gemma.rs,gemma/worker.rs,qwen.rs,qwen/adapters.rs,qwen/embedding.rs,qwen/lattice_worker.rs,qwen/vision.rs,qwen/worker.rs}`) stay disjoint from
/// `model/` and `forward/metal_*`; full path resolution runs first.
struct LocalItemNames {
    names: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for LocalItemNames {
    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        self.names.insert(node.ident.to_string());
        syn::visit::visit_item_enum(self, node);
    }

    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        self.names.insert(node.ident.to_string());
        syn::visit::visit_item_struct(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.names.insert(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        self.names.insert(node.ident.to_string());
        syn::visit::visit_item_trait(self, node);
    }

    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        self.names.insert(node.ident.to_string());
        syn::visit::visit_item_type(self, node);
    }

    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        self.names.insert(node.ident.to_string());
        syn::visit::visit_item_const(self, node);
    }

    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        self.names.insert(node.ident.to_string());
        syn::visit::visit_item_static(self, node);
    }

    fn visit_item_union(&mut self, node: &'ast syn::ItemUnion) {
        self.names.insert(node.ident.to_string());
        syn::visit::visit_item_union(self, node);
    }
}

fn collect_local_item_names(file: &syn::File) -> BTreeSet<String> {
    let mut collector = LocalItemNames {
        names: BTreeSet::new(),
    };
    collector.visit_file(file);
    collector.names
}

/// Multi-segment `Ident (:: Ident)+` runs in a macro invocation's raw
/// token stream, via the stream's own `Display` -- this crate has no
/// direct dependency on `proc_macro2`, so its `TokenTree` enum cannot be
/// named here to match on directly; `to_string()` is a trait method call
/// that needs no such naming. `proc_macro2`'s printer renders a `::`
/// (two `Punct(':')` tokens, the first `Joint`) as one glued, whitespace-
/// delimited `::` substring, so a run is exactly a maximal whitespace-
/// split chain `ident (:: ident)+`. Bare single-`Ident` heads (a local
/// binding, in this position almost always a `select!` arm's own pattern
/// name) are deliberately excluded -- see the module doc's sixth
/// narrowing for what token-shaped, non-syntactic coverage still misses.
fn path_shaped_token_runs(mac: &syn::Macro) -> Vec<Vec<String>> {
    fn is_plain_ident(word: &str) -> bool {
        let mut chars = word.chars();
        matches!(chars.next(), Some(c) if c.is_alphabetic() || c == '_')
            && chars.all(|c| c.is_alphanumeric() || c == '_')
    }
    let text = mac.tokens.to_string();
    // `proc_macro2`'s printer does not always separate a bracket/comma
    // from an adjacent identifier with a space -- a fresh call's first
    // argument prints as `new (std :: io :: ...`, where `(std` is ONE
    // whitespace-delimited word, gluing away exactly the leading segment
    // `is_plain_ident` would otherwise see (measured: `std::io::Error::
    // new(std::io::ErrorKind::BrokenPipe)`'s SECOND `std` was silently
    // dropped this way). Padding every bracket/comma/semicolon with
    // spaces before splitting makes each one its own word unconditionally,
    // so it can never glue to a neighboring identifier.
    let padded: String = text
        .chars()
        .flat_map(|c| match c {
            '(' | ')' | '{' | '}' | '[' | ']' | ',' | ';' => vec![' ', c, ' '],
            other => vec![other],
        })
        .collect();
    let words: Vec<&str> = padded.split_whitespace().collect();
    let mut runs = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if is_plain_ident(words[i]) {
            let mut segments = vec![words[i].to_string()];
            let mut j = i + 1;
            while j + 1 < words.len() && words[j] == "::" && is_plain_ident(words[j + 1]) {
                segments.push(words[j + 1].to_string());
                j += 2;
            }
            if segments.len() >= 2 {
                runs.push(segments);
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    runs
}

struct RawFnReturn {
    namespace: String,
    module_path: Vec<String>,
    key: String,
    raw: Vec<String>,
}

/// Two-phase index build. Phase 1 (this loop) collects every definition
/// and every namespace's set of discovered module paths, plus raw `use`
/// leaves and raw fn-return segments deferred for later. Phase 2 (below)
/// finalizes those deferred entries once every namespace's definitions and
/// module-path set are complete -- see `finalize_use_leaf`'s doc comment
/// for why this can't be done in one pass.
fn build_index_with_paths(
    walked: &BTreeMap<PathBuf, WalkedFile>,
    extra: &[(PathBuf, WalkedFile)],
    ctx: &CrossingCrateContext,
) -> BTreeMap<String, NamespaceIndex> {
    build_index_with_contexts(walked, extra, |_| ctx)
}

fn build_index_with_contexts<'a>(
    walked: &BTreeMap<PathBuf, WalkedFile>,
    extra: &[(PathBuf, WalkedFile)],
    context_for: impl Fn(&str) -> &'a CrossingCrateContext,
) -> BTreeMap<String, NamespaceIndex> {
    let mut indices: BTreeMap<String, NamespaceIndex> = BTreeMap::new();
    let mut raw_use_leaves: Vec<RawUseLeaf> = Vec::new();
    let mut raw_fn_returns: Vec<RawFnReturn> = Vec::new();

    for (path, walked_file) in walked.iter().chain(extra.iter().map(|(p, w)| (p, w))) {
        let index = indices.entry(walked_file.namespace.clone()).or_default();
        index.module_paths.insert(walked_file.module_path.clone());
        // Flattened rather than a plain `&walked_file.file.items` loop so
        // that a struct/enum/fn/const/etc. declared physically inside an
        // inline `mod name { ... }` block is indexed at ITS OWN module
        // path, not invisible to indexing the way a flat top-level-only
        // scan would leave it (`fn flatten_module_items`'s doc comment
        // covers the file-based-vs-inline distinction).
        let mut flattened_items = Vec::new();
        flatten_module_items(
            &walked_file.file.items,
            &walked_file.module_path,
            &mut flattened_items,
        );
        for (item_module_path, item) in &flattened_items {
            if item_module_path != &walked_file.module_path {
                index.module_paths.insert(item_module_path.clone());
            }
            match item {
                Item::Struct(s) => {
                    index.definitions.insert(
                        join_path(item_module_path, &s.ident.to_string()),
                        path.clone(),
                    );
                }
                Item::Enum(e) => {
                    index.definitions.insert(
                        join_path(item_module_path, &e.ident.to_string()),
                        path.clone(),
                    );
                }
                Item::Union(u) => {
                    index.definitions.insert(
                        join_path(item_module_path, &u.ident.to_string()),
                        path.clone(),
                    );
                }
                Item::Trait(t) => {
                    index.definitions.insert(
                        join_path(item_module_path, &t.ident.to_string()),
                        path.clone(),
                    );
                }
                Item::Type(t) => {
                    index.definitions.insert(
                        join_path(item_module_path, &t.ident.to_string()),
                        path.clone(),
                    );
                }
                Item::Const(c) => {
                    index.definitions.insert(
                        join_path(item_module_path, &c.ident.to_string()),
                        path.clone(),
                    );
                }
                Item::Static(s) => {
                    index.definitions.insert(
                        join_path(item_module_path, &s.ident.to_string()),
                        path.clone(),
                    );
                }
                Item::Fn(f) => {
                    let key = join_path(item_module_path, &f.sig.ident.to_string());
                    index.definitions.insert(key.clone(), path.clone());
                    if let Some(segments) = direct_return_segments(&f.sig.output) {
                        raw_fn_returns.push(RawFnReturn {
                            namespace: walked_file.namespace.clone(),
                            module_path: item_module_path.clone(),
                            key,
                            raw: segments,
                        });
                    }
                }
                _ => {}
            }
        }
        // `use` items are collected separately, via a full recursive visit
        // rather than the flat items loop above: a real, common idiom in
        // this tree scopes its imports to a single function body (`fn
        // run() { use lattice_inference::model::...; ... }`), which no
        // flat items scan ever reaches. Every `use` found anywhere in the
        // file -- module level, nested arbitrarily deep inside inline
        // modules, or nested arbitrarily deep inside function bodies -- is
        // attributed to its own innermost enclosing module's alias table
        // (`collect_use_items_with_module_path`'s doc comment). Within a
        // single module (inline or file-level) this is the "module-local"
        // reading the resolution algorithm's own wording asks for, not
        // block/function scoping: a deliberate, bounded approximation whose
        // residual risk is a same-module alias-name collision between two
        // functions each importing a DIFFERENT item under the SAME local
        // name (the later one wins). That is judged acceptable for a
        // resolver whose job is catching a forbidden dependency, not full
        // Rust name resolution.
        for (use_module_path, item_use) in
            collect_use_items_with_module_path(&walked_file.file, &walked_file.module_path)
        {
            let mut leaves = Vec::new();
            let mut globs = Vec::new();
            flatten_use_tree(&item_use.tree, &mut Vec::new(), &mut leaves, &mut globs);
            for (raw, local_name) in leaves {
                raw_use_leaves.push(RawUseLeaf {
                    namespace: walked_file.namespace.clone(),
                    module_path: use_module_path.clone(),
                    raw,
                    local_name,
                    is_pub: matches!(item_use.vis, syn::Visibility::Public(_)),
                });
            }
            // A glob's prefix resolves the same way `canonicalize_use_target`
            // resolves any `crate`/`self`/`super`-rooted path: no second
            // index pass is needed, unlike a NAMED leaf's unqualified first
            // segment (`finalize_use_leaf`'s uniform-paths check), because
            // there is no final item name here to disambiguate against an
            // extern crate -- only `crate`/`self`/`super`-prefixed globs are
            // resolved; an external-crate-prefixed glob (`use tokio::sync::
            // *;`) is left unrecorded, a documented conservative gap (see
            // `flatten_use_tree`'s glob-arm comment).
            for prefix in globs {
                if let Some(resolved) = canonicalize_use_target(&use_module_path, &prefix) {
                    index
                        .glob_imports
                        .entry(use_module_path.clone())
                        .or_default()
                        .push(resolved);
                }
            }
        }
    }

    for leaf in raw_use_leaves {
        let Some(resolved) = finalize_use_leaf(
            context_for(&leaf.namespace),
            &indices,
            &leaf.namespace,
            &leaf.module_path,
            &leaf.raw,
        ) else {
            continue;
        };
        let index = indices.entry(leaf.namespace.clone()).or_default();
        index.aliases.insert(
            (leaf.module_path.join("::"), leaf.local_name.clone()),
            resolved.clone(),
        );
        if leaf.is_pub {
            index
                .reexports
                .insert(join_path(&leaf.module_path, &leaf.local_name), resolved);
        }
    }

    for entry in raw_fn_returns {
        if let Some(resolved) = finalize_use_leaf(
            context_for(&entry.namespace),
            &indices,
            &entry.namespace,
            &entry.module_path,
            &entry.raw,
        ) {
            indices
                .entry(entry.namespace)
                .or_default()
                .fn_returns
                .insert(entry.key, resolved);
        }
    }

    indices
}

// ---------------------------------------------------------------------
// Controls
//
// Each fixture lives under
// `tests/data/pipeline_boundary_fixtures/<case>/` and is analyzed as its
// own tiny crate-less tree (`crate_dir` = the fixture directory itself,
// which has no `Cargo.toml`, so every root collapses into the single
// `"self"` namespace -- see the module doc comment). Every rejection
// fixture also exercises a neutral, non-origin reference in the SAME
// frontend file, so a resolver that rejected everything indiscriminately
// would fail that half of the assertion.
// ---------------------------------------------------------------------

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pipeline_boundary_fixtures")
        .join(name)
}

fn resolved_paths(report: &BoundaryReport) -> Vec<String> {
    report
        .crossings
        .iter()
        .map(|crossing| crossing.resolved_path.clone())
        .collect()
}

fn assert_module_path_fixture(files: &[(&str, &str, &[&str])]) {
    let fixture = tempfile::tempdir().expect("create module path fixture");
    for (relative, source, _) in files {
        let path = fixture.path().join(relative);
        std::fs::create_dir_all(path.parent().expect("fixture source parent"))
            .expect("create fixture source directory");
        std::fs::write(path, source).expect("write fixture source");
    }
    let mut walker = Walker::new(fixture.path(), true).expect("create module walker");
    walker
        .walk_root(&fixture.path().join("src/lib.rs"), "fixture", Vec::new())
        .expect("resolve compiler-selected module closure");
    let actual = walker
        .files
        .into_iter()
        .map(|(path, file)| (path, (file.namespace, file.module_path)))
        .collect::<BTreeMap<_, _>>();
    let expected = files
        .iter()
        .map(|(relative, _, module_path)| {
            (
                std::fs::canonicalize(fixture.path().join(relative))
                    .expect("canonicalize fixture source"),
                (
                    "fixture".to_string(),
                    module_path.iter().map(|name| (*name).to_string()).collect(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(actual, expected);
}

#[test]
fn module_path_attribute_in_non_mod_rs_uses_source_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "mod a;", &[]),
        ("src/a.rs", "#[path = \"a/t.rs\"] mod t;", &["a"]),
        ("src/a/t.rs", "", &["a", "t"]),
    ]);
}

#[test]
fn module_path_attribute_in_crate_root_uses_source_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "#[path = \"nested/t.rs\"] mod t;", &[]),
        ("src/nested/t.rs", "", &["t"]),
    ]);
}

#[test]
fn module_path_attribute_in_mod_rs_uses_source_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "mod a;", &[]),
        ("src/a/mod.rs", "#[path = \"t.rs\"] mod t;", &["a"]),
        ("src/a/t.rs", "", &["a", "t"]),
    ]);
}

#[test]
fn module_path_attribute_file_children_use_containing_directory() {
    assert_module_path_fixture(&[
        (
            "src/lib.rs",
            "#[path = \"nested/alternate.rs\"] mod renamed;",
            &[],
        ),
        ("src/nested/alternate.rs", "mod child;", &["renamed"]),
        (
            "src/nested/child.rs",
            "mod grandchild;",
            &["renamed", "child"],
        ),
        (
            "src/nested/child/grandchild.rs",
            "",
            &["renamed", "child", "grandchild"],
        ),
    ]);
}

#[test]
fn module_path_attribute_inside_non_mod_rs_inline_uses_module_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "mod a;", &[]),
        (
            "src/a.rs",
            "mod inline { #[path = \"other.rs\"] mod inner; }",
            &["a"],
        ),
        ("src/a/inline/other.rs", "", &["a", "inline", "inner"]),
    ]);
}

#[test]
fn module_path_attribute_inside_root_inline_uses_module_directory() {
    assert_module_path_fixture(&[
        (
            "src/lib.rs",
            "mod inline { #[path = \"other.rs\"] mod inner; }",
            &[],
        ),
        ("src/inline/other.rs", "", &["inline", "inner"]),
    ]);
}

#[test]
fn module_path_attribute_file_inline_children_use_containing_directory() {
    assert_module_path_fixture(&[
        (
            "src/lib.rs",
            "#[path = \"nested/alternate.rs\"] mod renamed;",
            &[],
        ),
        (
            "src/nested/alternate.rs",
            "mod inline { #[path = \"other.rs\"] mod inner; }",
            &["renamed"],
        ),
        (
            "src/nested/inline/other.rs",
            "",
            &["renamed", "inline", "inner"],
        ),
    ]);
}

/// MUST BE REJECTED: `use crate::model::ConcreteModel as Renamed;` followed
/// by a reference through the alias name alone. Passing requires the
/// alias table (`finalize_use_leaf` / the single-segment alias lookup in
/// `Resolver::resolve_reference`), not just recognizing the `use` line's
/// own raw path.
#[test]
fn renamed_import_alias_is_rejected() {
    let dir = fixture_dir("renamed_import_alias");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[
            dir.join("model/mod.rs"),
            dir.join("neutral/mod.rs"),
            dir.join("alias_owner/mod.rs"),
        ],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    // Cross-module arm (Decision B): if `Resolver::lookup`'s alias-table
    // chase did not fire for `crate::alias_owner::Renamed`, that
    // reference cannot resolve any other way (it is not a `pub use`, so
    // `reexports` never sees it; `LocalItemNames` never captures a `use`
    // alias either) -- it would land here as a genuine unresolved entry,
    // not silently disappear.
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.kind == "path" && c.resolved_path.ends_with("model::ConcreteModel")),
        "the alias reference itself must resolve to the origin definition (either arm): {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin must not be reported (either arm): {:?}",
        report.crossings
    );
}

#[test]
fn inline_module_nesting_reports_crossing() {
    let dir = fixture_dir("inline_module_nesting");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[dir.join("model/mod.rs"), dir.join("neutral/mod.rs")],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.kind == "path" && c.resolved_path.ends_with("model::ConcreteModel")),
        "a reference nested inside an inline `mod` must still resolve to the origin \
         definition: {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin at the same nesting depth must not be reported: {:?}",
        report.crossings
    );
}

/// MUST BE REPORTED: an origin-type reference written inside a
/// `tokio::select!`-shaped macro body -- opaque to syn's parsed-item/
/// parsed-expr arms, since a function-like macro's arguments are never
/// parsed into structured items or exprs. Passing requires the
/// token-shaped fallback (`path_shaped_token_runs`, fed through the same
/// `record` the parsed-path arm uses). `consume_neutral`'s
/// `PlainOptions::from_directory` is the negative twin: identical macro
/// shape and nesting depth, non-origin reference, must not be reported.
#[test]
fn macro_token_scan_reports_origin_reference() {
    let dir = fixture_dir("macro_token_scan");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[dir.join("model/mod.rs"), dir.join("neutral/mod.rs")],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.kind == "macro-token" && c.resolved_path.ends_with("model::ConcreteModel")),
        "the token-shaped scan must report the origin reference inside the macro body: {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin must not be reported: {:?}",
        report.crossings
    );
}

/// MUST BE REJECTED: a `pub use` chain (`reexport/mod.rs` re-exports
/// `model::ConcreteModel`; `frontend.rs` imports it from `reexport`, not
/// from `model` directly). Passing requires chasing `reexports` in
/// `Resolver::lookup`, and the reported item must still be the true
/// defining path, not the reexport hop.
#[test]
fn reexport_chain_is_rejected() {
    let dir = fixture_dir("reexport_chain");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[
            dir.join("model/mod.rs"),
            dir.join("neutral/mod.rs"),
            dir.join("reexport/mod.rs"),
        ],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.resolved_path.ends_with("model::ConcreteModel")),
        "the re-export chain must resolve to the true defining item: {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin must not be reported: {:?}",
        report.crossings
    );
}

/// MUST BE REJECTED: `factory::build() -> crate::model::ConcreteModel` is
/// never itself scanned (it is an index-only root); the only way its
/// result reaches the report is the bounded `let x = f(..)` inferred-
/// constructor rule. `factory::build_neutral()` is the negative twin: same
/// mechanism, non-origin return type, must not be reported.
#[test]
fn inferred_constructor_is_rejected() {
    let dir = fixture_dir("inferred_constructor");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[
            dir.join("model/mod.rs"),
            dir.join("neutral/mod.rs"),
            dir.join("factory/mod.rs"),
        ],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.kind == "inferred-constructor"
                && c.resolved_path.ends_with("model::ConcreteModel")),
        "the inferred-constructor rule must connect build() to its origin return type: {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin (build_neutral) must not be reported: {:?}",
        report.crossings
    );
}

/// MUST BE REJECTED: the same `factory::build()` as above, called through
/// `facade`'s `pub use` re-export. The callee's key names the re-exporting
/// module, which defines no function, so the rule has to follow the
/// re-export to `factory` before it can read the declared return type.
/// `facade::build_neutral()` is the negative twin.
#[test]
fn inferred_constructor_through_a_reexport_is_rejected() {
    let dir = fixture_dir("inferred_constructor_reexport");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[
            dir.join("model/mod.rs"),
            dir.join("neutral/mod.rs"),
            dir.join("factory/mod.rs"),
            dir.join("facade/mod.rs"),
        ],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.kind == "inferred-constructor"
                && c.resolved_path.ends_with("model::ConcreteModel")),
        "the inferred-constructor rule must follow facade::build to factory::build: {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin (build_neutral) must not be reported: {:?}",
        report.crossings
    );
}

/// MUST BE REJECTED: `crate::model::ConcreteModel` appears only as a
/// generic argument (`Wrapper<ConcreteModel>`), never as the head of its
/// own reference. Passing requires nothing beyond syn's own recursive
/// `Visit` routing generic arguments back through `visit_path` -- no
/// special-casing of the wrapper type. `Wrapper<PlainOptions>` is the
/// negative twin.
#[test]
fn generic_wrapper_is_rejected() {
    let dir = fixture_dir("generic_wrapper");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[dir.join("model/mod.rs"), dir.join("neutral/mod.rs")],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.resolved_path.ends_with("model::ConcreteModel")),
        "a generic argument must be reached the same as a bare reference: {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin (Wrapper<PlainOptions>) must not be reported: {:?}",
        report.crossings
    );
}

/// MUST BE REJECTED (or explicitly unresolved): `crate::model::ConcreteModel`
/// is passed only as opaque tokens to an unrecognized macro invocation, so
/// there is no `syn::Path` reaching `visit_path` at all -- the ONLY sound
/// way to reject this is the macro-completeness check
/// (`unclassifiable_item_macro`) failing the run. This fixture also
/// defines a local `macro_rules! harmless { .. }` purely as a regression
/// check: `macro_rules!` items are themselves represented in syn as a
/// macro invocation of the literal name `macro_rules` (see
/// `CrossingScanner::visit_item_macro`'s doc comment), so a scanner that
/// let the default `visit_item_macro` recursion run would wrongly report
/// every macro-DEFINING file as unclassifiable too.
#[test]
fn macro_hidden_construction_is_rejected_or_unresolved() {
    let dir = fixture_dir("macro_hidden_construction");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[dir.join("model/mod.rs"), dir.join("neutral/mod.rs")],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.crossings.is_empty(),
        "a macro-hidden construction must never surface as a resolved crossing: {:?}",
        report.crossings
    );
    assert!(
        report
            .unresolved
            .iter()
            .any(|entry| entry.contains("some_unclassified_macro")),
        "the hidden construction must fail completeness instead: {:?}",
        report.unresolved
    );
    assert!(
        !report
            .unresolved
            .iter()
            .any(|entry| entry.contains("macro_rules")),
        "a macro_rules! DEFINITION must never itself be flagged as an unclassifiable invocation: {:?}",
        report.unresolved
    );
}

/// MUST BE REJECTED: the frontend root (`entry/frontend.rs`) reaches
/// `backdoor.rs`, which physically lives OUTSIDE the frontend root's own
/// directory, via `#[path = "../backdoor.rs"] mod backdoor;`. Passing
/// requires frontend membership to be a module-path-prefix test (as the
/// module doc comment states), not a directory-prefix test, and
/// `backdoor.rs`'s own origin/neutral references must still be classified
/// correctly once discovered.
#[test]
fn sibling_via_path_is_rejected() {
    let dir = fixture_dir("sibling_via_path");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("entry/frontend.rs")],
        &[dir.join("model/mod.rs"), dir.join("neutral/mod.rs")],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.unresolved.is_empty(),
        "unexpected unresolved: {:?}",
        report.unresolved
    );
    assert!(
        report
            .modules_read
            .iter()
            .any(|path| path.ends_with("backdoor.rs")),
        "the #[path]-relocated sibling must be discovered as frontend: {:?}",
        report.modules_read
    );
    assert!(
        report
            .crossings
            .iter()
            .any(|c| c.resolved_path.ends_with("model::ConcreteModel")),
        "the relocated sibling's own origin reference must still be reported: {:?}",
        report.crossings
    );
    assert!(
        !resolved_paths(&report)
            .iter()
            .any(|p| p.contains("PlainOptions")),
        "the neutral negative twin must not be reported: {:?}",
        report.crossings
    );
}

/// MUST PASS: the only mentions of the origin type are inside a `//`
/// comment and a string literal -- neither is ever represented as a
/// `syn::Path`, so a clean run with zero crossings and zero unresolved
/// entries is the only correct answer. This is also what makes the
/// comment/string case pass "for free": nothing about it needs handling,
/// it is simply never seen by `visit_path`.
#[test]
fn comment_and_string_decoy_passes_clean() {
    let dir = fixture_dir("comment_string_decoy");
    let report = analyze_boundary(
        &dir,
        &["model"],
        &[dir.join("frontend.rs")],
        &[dir.join("model/mod.rs")],
        false,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        report.crossings.is_empty(),
        "a comment/string mention must never be treated as a reference: {:?}",
        report.crossings
    );
    assert!(
        report.unresolved.is_empty(),
        "a comment/string mention must never fail completeness: {:?}",
        report.unresolved
    );
}

/// MUST FAIL: the frontend root's bytes are not valid UTF-8. Built at test
/// run time via `tempfile::tempdir()` (matching the shared support
/// module's own fixture convention) rather than as a checked-in file,
/// since a public repository is not the right home for a deliberately
/// invalid byte sequence. The negative twin proves the SAME call shape
/// with valid bytes succeeds, so the failure is attributable to the bad
/// bytes specifically.
#[test]
fn unreadable_source_fails() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let bad_path = dir.path().join("frontend.rs");
    let mut bytes = b"pub fn f() {}\n".to_vec();
    bytes.push(0xFF); // never valid as a UTF-8 continuation or lead byte
    std::fs::write(&bad_path, &bytes).expect("write invalid-UTF-8 fixture");

    let result = analyze_boundary(dir.path(), &[], &[bad_path], &[], false);
    let Err(reason) = result else {
        panic!("expected an error reading invalid UTF-8, got Ok");
    };
    assert!(
        reason.contains("could not read"),
        "unexpected error shape: {reason}"
    );
}

#[test]
fn unreadable_source_negative_twin_succeeds() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let good_path = dir.path().join("frontend.rs");
    std::fs::write(&good_path, b"pub fn f() {}\n").expect("write valid fixture");

    let report = analyze_boundary(dir.path(), &[], &[good_path], &[], false)
        .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(report.crossings.is_empty());
    assert!(report.unresolved.is_empty());
}

/// MUST FAIL: an empty frontend-root set must trip `analyze_boundary`'s
/// own "an empty root set is a failure, not a clean run" guard -- the
/// mechanism the production test's known-positive-roots assertion relies
/// on. The negative twin proves the guard does not ALSO fire when a root
/// genuinely is supplied (a rule that rejects an empty set by rejecting
/// every set would still pass a naive "it failed" assertion).
#[test]
fn missing_frontend_root_fails() {
    let dir = fixture_dir("missing_frontend_root");
    let result = analyze_boundary(&dir, &[], &[], &[], false);
    let Err(reason) = result else {
        panic!("expected an error for an empty frontend-root set, got Ok");
    };
    assert!(
        reason.contains("no frontend files were discovered"),
        "unexpected error shape: {reason}"
    );
}

#[test]
fn missing_frontend_root_negative_twin_succeeds() {
    let dir = fixture_dir("missing_frontend_root");
    let report = analyze_boundary(&dir, &[], &[dir.join("frontend.rs")], &[], false)
        .unwrap_or_else(|reason| panic!("{reason}"));
    assert_eq!(report.modules_read.len(), 1);
    assert!(report.crossings.is_empty());
    assert!(report.unresolved.is_empty());
}

fn pipeline_package_fixture() -> tempfile::TempDir {
    let fixture = tempfile::tempdir().expect("pipeline package fixture");
    for (package, name) in [
        ("inference", "lattice-inference"),
        ("serve", "lattice-serve"),
    ] {
        let root = fixture.path().join(package);
        std::fs::create_dir_all(root.join("src/bin")).expect("fixture src/bin");
        let dependency = if package == "serve" {
            "[dependencies]\nlattice-inference = { path = '../inference' }\n"
        } else {
            ""
        };
        std::fs::write(root.join("Cargo.toml"), format!(
            "[package]\nname = '{name}'\nversion = '0.0.0'\nedition = '2024'\n[workspace]\n{dependency}"
        )).expect("fixture manifest");
        std::fs::write(
            root.join("src/lib.rs"),
            if package == "inference" {
                "pub mod model; pub mod pipeline { pub struct Allowed; }"
            } else {
                "pub fn frontend() {}"
            },
        )
        .expect("fixture library");
        std::fs::write(root.join("src/bin/frontend.rs"), if package == "serve" {
            "use lattice_inference::model::ConcreteModel as Renamed;\nfn main() { let _ = Renamed; let _ = lattice_inference::pipeline::Allowed; }"
        } else { "fn main() { let _ = lattice_inference::pipeline::Allowed; }" }).expect("fixture bin");
    }
    std::fs::create_dir_all(fixture.path().join("inference/src/model")).expect("fixture model");
    std::fs::write(
        fixture.path().join("inference/src/model/mod.rs"),
        "pub struct ConcreteModel;",
    )
    .expect("fixture model source");
    fixture
}

#[test]
fn pipeline_production_collector_reports_second_package_crossing() {
    let fixture = pipeline_package_fixture();
    let packages = scanned_packages_at(&fixture.path().join("inference"), &[".", "../serve"])
        .expect("fixture packages");
    let population = BoundaryPopulation {
        origins: &[(".", "src/model")],
        frontends: &[],
        providers: &[],
        known_positive: &[
            (".", "src/bin/frontend.rs"),
            ("../serve", "src/bin/frontend.rs"),
        ],
    };
    let report = collect_pipeline_boundary(&packages, &population).expect("fixture population");
    assert!(report.unresolved.is_empty(), "{:?}", report.unresolved);
    assert!(
        report.crossings.iter().any(|crossing| {
            crossing.relative_path == "../serve:src/bin/frontend.rs"
                && crossing.resolved_path == ".:crate::model::ConcreteModel"
        }),
        "second-package alias must resolve to the inference-owned origin: {:?}",
        report.crossings
    );
    assert!(
        !report
            .crossings
            .iter()
            .any(|crossing| crossing.resolved_path.contains("Allowed")),
        "the allowed pipeline reference must remain outside the origin classification"
    );
    assert!(report.modules_read.contains(".:src/bin/frontend.rs"));
    assert!(report.modules_read.contains("../serve:src/bin/frontend.rs"));
    assert_eq!(
        report.modules_read.len(),
        2,
        "same-relativepath package keys must remain distinct"
    );
    assert_eq!(
        report
            .targets
            .iter()
            .filter(|(_, kind, _)| kind == "bin")
            .count(),
        2
    );
    assert!(
        report
            .targets
            .iter()
            .any(|(_, _, path)| path == ".:src/lib.rs")
    );
    assert!(
        report
            .targets
            .iter()
            .any(|(_, _, path)| path == "../serve:src/lib.rs")
    );
}

#[test]
fn pipeline_production_collector_resolves_renamed_scanned_dependency() {
    let fixture = pipeline_package_fixture();
    let serve = fixture.path().join("serve");
    let manifest = serve.join("Cargo.toml");
    let source = serve.join("src/bin/frontend.rs");
    std::fs::write(
        &manifest,
        std::fs::read_to_string(&manifest)
            .expect("fixture manifest")
            .replace(
                "lattice-inference = { path = '../inference' }",
                "engine = { package = 'lattice-inference', path = '../inference' }",
            ),
    )
    .expect("renamed dependency manifest");
    std::fs::write(
        &source,
        std::fs::read_to_string(&source)
            .expect("fixture source")
            .replace("lattice_inference::", "engine::"),
    )
    .expect("renamed dependency source");
    let packages = scanned_packages_at(&fixture.path().join("inference"), &[".", "../serve"])
        .expect("fixture packages");
    let population = BoundaryPopulation {
        origins: &[(".", "src/model")],
        frontends: &[],
        providers: &[],
        known_positive: &[],
    };
    let report = collect_pipeline_boundary(&packages, &population).expect("fixture population");
    assert!(report.unresolved.is_empty(), "{:?}", report.unresolved);
    assert!(
        report.crossings.iter().any(|crossing| {
            crossing.relative_path == "../serve:src/bin/frontend.rs"
                && crossing.resolved_path == ".:crate::model::ConcreteModel"
        }),
        "a renamed scanned dependency must not be classified as external"
    );
}

#[test]
fn pipeline_production_collector_resolves_reexported_model_alias() {
    let fixture = pipeline_package_fixture();
    let inference = fixture.path().join("inference");
    let serve = fixture.path().join("serve");
    std::fs::write(
        inference.join("src/lib.rs"),
        "pub mod model; pub use model::ConcreteModel as PublicModel; pub mod pipeline { pub struct Allowed; }",
    )
    .expect("fixture library re-export");
    std::fs::write(
        serve.join("src/bin/frontend.rs"),
        "use lattice_inference::PublicModel as Renamed; fn main() { let _ = Renamed; let _ = lattice_inference::pipeline::Allowed; }",
    )
    .expect("fixture re-export consumer");
    let packages = scanned_packages_at(&inference, &[".", "../serve"]).expect("fixture packages");
    let population = BoundaryPopulation {
        origins: &[(".", "src/model")],
        frontends: &[],
        providers: &[],
        known_positive: &[("../serve", "src/bin/frontend.rs")],
    };
    let report = collect_pipeline_boundary(&packages, &population).expect("fixture population");
    assert!(report.unresolved.is_empty(), "{:?}", report.unresolved);
    assert!(
        report.crossings.iter().any(|crossing| {
            crossing.relative_path == "../serve:src/bin/frontend.rs"
                && crossing.resolved_path == ".:crate::model::ConcreteModel"
        }),
        "the consumer alias must follow the library re-export into the model origin: {:?}",
        report.crossings
    );
    assert!(
        !report
            .crossings
            .iter()
            .any(|crossing| crossing.resolved_path.contains("Allowed")),
        "the allowed pipeline neighbor must remain outside the model origin"
    );
}

#[test]
fn pipeline_baseline_preserves_package_qualified_keys() {
    let original = parse_baseline("src/bin/frontend.rs\tcrate::model::ConcreteModel\n");
    assert_eq!(
        original,
        BTreeSet::from([(
            ".:src/bin/frontend.rs".to_string(),
            ".:crate::model::ConcreteModel".to_string(),
        )])
    );
    let mut two_packages = original;
    two_packages.insert((
        "../serve:src/bin/frontend.rs".into(),
        ".:crate::model::ConcreteModel".into(),
    ));
    assert_eq!(
        parse_baseline(&format_baseline(&two_packages)),
        two_packages
    );
    assert_eq!(two_packages.len(), 2);
}

// ---------------------------------------------------------------------
// Production analysis
// ---------------------------------------------------------------------

/// The production run scans `src/serve`, the provider roots, `src/serving_worker.rs`, and every
/// binary against the baseline. A ratchet, not a migration verdict: a new crossing, or a stale
/// baseline entry for one that no longer exists, fails the test; a nonzero count does not.
#[test]
fn pipeline_boundary_matches_recorded_baseline() {
    let report = collect_pipeline_boundary(&scanned_packages(), &PRODUCTION_POPULATION)
        .unwrap_or_else(|reason| panic!("{reason}"));

    eprintln!("cargo targets selected:");
    for (name, kind, path) in &report.targets {
        eprintln!("  {name} ({kind}) {path}");
    }
    eprintln!("modules discovered: {}", report.modules_read.len());
    eprintln!("files read (frontend): {}", report.modules_read.len());
    for crossing in &report.crossings {
        eprintln!(
            "crossing: {} -> {} [{}]",
            crossing.relative_path, crossing.resolved_path, crossing.kind
        );
    }
    eprintln!("pipeline boundary crossings: {}", report.crossings.len());

    assert!(
        report.unresolved.is_empty(),
        "unresolved references are a completeness failure: {:?}",
        report.unresolved
    );

    let discovered = baseline_lines(&report);
    let baseline_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pipeline_boundary_baseline.txt");
    let baseline_text = std::fs::read_to_string(&baseline_path)
        .unwrap_or_else(|reason| panic!("could not read {}: {reason}", baseline_path.display()));
    let recorded = parse_baseline(&baseline_text);

    let added: Vec<_> = discovered.difference(&recorded).collect();
    let removed: Vec<_> = recorded.difference(&discovered).collect();
    assert!(
        added.is_empty() && removed.is_empty(),
        "pipeline boundary population changed since the recorded baseline.\n\
         added (this PR introduced a new crossing): {added:#?}\n\
         removed (a migration happened; delete these lines from {}): {removed:#?}\n\
         the full current population, to replace the file's contents with if this is a \
         genuine migration:\n{}",
        baseline_path.display(),
        format_baseline(&discovered)
    );
}
