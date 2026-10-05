//! Environment-variable contract for every `LATTICE_*` name the non-test source reads.
//!
//! Two properties, both decided on the parsed syntax tree rather than on text, because
//! rustfmt splits `var_os(..).is_some()` chains across lines and a comment that names a
//! variable must not satisfy either check:
//!
//! (a) No non-test source under `crates/*/src/` reads a `LATTICE_*` variable by presence
//!     (`NAME=0` would then enable the feature) unless the name is on
//!     `PRESENCE_ALLOWLIST` with a reason. Switches go through `crate::env_switch_enabled`.
//! (b) The set of `LATTICE_*` names that non-test source under `crates/*/src/` names
//!     equals the set of rows in `crates/inference/CONFIG.md`'s variable table.
//!
//! "Non-test" means: reachable from a crate root with `cfg(test)` false, and not inside an
//! item carrying `#[test]` or a `cfg` that holds only when `test` does. Variables named
//! only by tests, benches or examples are outside the table by design.
//!
//! Residual gaps, stated rather than hidden: a presence check spelled as a `match` on the
//! lookup, or as `matches!`, or through a helper that takes the name as a runtime value,
//! or by handing `std::env::var_os` itself to `map`, is not recognised by (a); an
//! unrecognised helper still has to name its variable with a string literal, so (b) sees
//! the name. `measurement.rs` reads its three GPU handoff variables that last way, as a
//! presence group set by the benchmark supervisor; CONFIG.md says so in their rows.

#[path = "support/source_graph.rs"]
#[allow(dead_code)]
mod source_graph;
use source_graph::{module_source_closure, rust_sources_under, syn_attributes_formula};

use proc_macro2::{TokenStream, TokenTree};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{Attribute, Expr, Meta, Token};

/// Presence-checked variables that are allowed, each with the reason it is.
const PRESENCE_ALLOWLIST: &[(&str, &str)] = &[(
    "LATTICE_METAL_TEST_ENFORCE",
    "test-only: any value, including 0, makes a missing Metal device a failure instead of a skip",
)];

const PRESENCE_METHODS: &[&str] = &["is_some", "is_none", "is_ok", "is_err"];
const UNRESOLVED_NAME: &str = "<unresolved>";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// The name appears as a string literal (constant, argument, attribute value).
    Literal,
    /// Read through `env_switch_enabled(name)`.
    Switch,
    /// Read through `env::var(name)` / `env::var_os(name)` with the value used.
    Value,
    /// Read through `env::var(name)` / `env::var_os(name)` and only its presence used.
    Presence,
}

struct Hit {
    name: String,
    kind: Kind,
    krate: String,
    file: String,
    scope: String,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve workspace root")
}

fn is_lattice_name(text: &str) -> bool {
    text.strip_prefix("LATTICE_").is_some_and(|rest| {
        !rest.is_empty()
            && rest
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
    })
}

/// True for an item that exists only in a test build: `#[test]`-style attributes, or a
/// `cfg` that is satisfiable with `test` true and not with `test` false. Features other
/// than `test` stay free, so `cfg(not(feature = ..))` code is still scanned.
fn test_only(attributes: &[Attribute]) -> bool {
    if attributes.iter().any(|attribute| {
        attribute
            .path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
    }) {
        return true;
    }
    let without = syn_attributes_formula(attributes, false, "env_var_contract")
        .and_then(|formula| formula.satisfiable())
        .expect("classify cfg without test");
    let with = syn_attributes_formula(attributes, true, "env_var_contract")
        .and_then(|formula| formula.satisfiable())
        .expect("classify cfg with test");
    with && !without
}

fn string_literal(token: &proc_macro2::Literal) -> Option<String> {
    match syn::Lit::new(token.clone()) {
        syn::Lit::Str(text) => Some(text.value()),
        _ => None,
    }
}

fn peel(mut expression: &Expr) -> &Expr {
    loop {
        match expression {
            Expr::Paren(inner) => expression = &inner.expr,
            Expr::Group(inner) => expression = &inner.expr,
            Expr::Reference(inner) => expression = &inner.expr,
            _ => return expression,
        }
    }
}

fn last_segment(path: &syn::Path) -> Option<String> {
    path.segments
        .last()
        .map(|segment| segment.ident.to_string())
}

/// `std::env::var` / `env::var_os` (or a bare imported `var` / `var_os`).
fn is_env_lookup(call: &syn::ExprCall) -> bool {
    let Expr::Path(function) = peel(&call.func) else {
        return false;
    };
    let Some(name) = last_segment(&function.path) else {
        return false;
    };
    if name != "var" && name != "var_os" {
        return false;
    }
    let qualified = function
        .path
        .segments
        .iter()
        .any(|segment| segment.ident == "env");
    qualified || function.path.segments.len() == 1
}

fn env_lookup_call(expression: &Expr) -> Option<&syn::ExprCall> {
    match peel(expression) {
        Expr::Call(call) if is_env_lookup(call) => Some(call),
        _ => None,
    }
}

/// Collects `const NAME: &str = "LATTICE_..."` so a lookup through the constant resolves.
#[derive(Default)]
struct ConstCollector {
    values: BTreeMap<String, BTreeSet<String>>,
}

impl ConstCollector {
    fn record(&mut self, ident: &syn::Ident, expression: &Expr) {
        if let Expr::Lit(literal) = peel(expression)
            && let syn::Lit::Str(text) = &literal.lit
        {
            self.values
                .entry(ident.to_string())
                .or_default()
                .insert(text.value());
        }
    }
}

impl<'ast> Visit<'ast> for ConstCollector {
    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        if !test_only(&node.attrs) {
            self.record(&node.ident, &node.expr);
        }
    }
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if !test_only(&node.attrs) {
            syn::visit::visit_item_mod(self, node);
        }
    }
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if !test_only(&node.attrs) {
            syn::visit::visit_item_fn(self, node);
        }
    }
    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if !test_only(&node.attrs) {
            syn::visit::visit_item_impl(self, node);
        }
    }
    fn visit_impl_item_const(&mut self, node: &'ast syn::ImplItemConst) {
        if !test_only(&node.attrs) {
            self.record(&node.ident, &node.expr);
        }
    }
}

struct Scanner<'a> {
    consts: &'a BTreeMap<String, BTreeSet<String>>,
    krate: &'a str,
    file: &'a str,
    scope: Vec<String>,
    hits: Vec<Hit>,
}

impl Scanner<'_> {
    fn record(&mut self, name: &str, kind: Kind) {
        self.hits.push(Hit {
            name: name.to_string(),
            kind,
            krate: self.krate.to_string(),
            file: self.file.to_string(),
            scope: if self.scope.is_empty() {
                "<file scope>".to_string()
            } else {
                self.scope.join("::")
            },
        });
    }

    /// Names a lookup's first argument can take: a literal, or a constant holding one.
    fn argument_names(&self, call: &syn::ExprCall) -> Vec<String> {
        let Some(argument) = call.args.first() else {
            return Vec::new();
        };
        match peel(argument) {
            Expr::Lit(literal) => match &literal.lit {
                syn::Lit::Str(text) => vec![text.value()],
                _ => Vec::new(),
            },
            Expr::Path(path) => last_segment(&path.path)
                .and_then(|ident| self.consts.get(&ident))
                .map(|values| values.iter().cloned().collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn record_lookup(&mut self, call: &syn::ExprCall, kind: Kind) {
        let names = self.argument_names(call);
        let lattice: Vec<_> = names.iter().filter(|name| is_lattice_name(name)).collect();
        if lattice.is_empty() {
            // A presence check on a name this scan cannot resolve fails closed: it
            // cannot be shown to be anything but a switch read by presence.
            if kind == Kind::Presence && names.is_empty() {
                self.record(UNRESOLVED_NAME, Kind::Presence);
            }
            return;
        }
        for name in lattice {
            self.record(name, kind);
        }
    }

    /// Token-level fallback for macro bodies that do not parse as comma-separated
    /// expressions (`?x`, `%x` fields and similar).
    fn scan_tokens(&mut self, tokens: TokenStream) {
        let flat: Vec<TokenTree> = tokens.into_iter().collect();
        for (index, tree) in flat.iter().enumerate() {
            match tree {
                TokenTree::Group(group) => self.scan_tokens(group.stream()),
                TokenTree::Literal(literal) => {
                    if let Some(text) = string_literal(literal)
                        && is_lattice_name(&text)
                    {
                        self.record(&text, Kind::Literal);
                    }
                }
                TokenTree::Ident(ident) if ident == "var" || ident == "var_os" => {
                    let Some(TokenTree::Group(arguments)) = flat.get(index + 1) else {
                        continue;
                    };
                    let tail_is_presence = matches!(
                        (flat.get(index + 2), flat.get(index + 3)),
                        (Some(TokenTree::Punct(dot)), Some(TokenTree::Ident(method)))
                            if dot.as_char() == '.'
                                && PRESENCE_METHODS.iter().any(|name| method == name)
                    );
                    if !tail_is_presence {
                        continue;
                    }
                    let name = arguments.stream().into_iter().find_map(|tree| match tree {
                        TokenTree::Literal(literal) => string_literal(&literal),
                        _ => None,
                    });
                    match name {
                        Some(text) if is_lattice_name(&text) => {
                            self.record(&text, Kind::Presence);
                        }
                        Some(_) => {}
                        None => self.record(UNRESOLVED_NAME, Kind::Presence),
                    }
                }
                _ => {}
            }
        }
    }

    fn scoped<T>(&mut self, label: String, body: impl FnOnce(&mut Self) -> T) -> T {
        self.scope.push(label);
        let result = body(self);
        self.scope.pop();
        result
    }
}

impl<'ast> Visit<'ast> for Scanner<'_> {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if !test_only(&node.attrs) {
            self.scoped(format!("mod {}", node.ident), |this| {
                syn::visit::visit_item_mod(this, node)
            });
        }
    }
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if !test_only(&node.attrs) {
            self.scoped(format!("fn {}", node.sig.ident), |this| {
                syn::visit::visit_item_fn(this, node)
            });
        }
    }
    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if !test_only(&node.attrs) {
            let label = match &*node.self_ty {
                syn::Type::Path(path) => last_segment(&path.path).unwrap_or_default(),
                _ => "impl".to_string(),
            };
            self.scoped(format!("impl {label}"), |this| {
                syn::visit::visit_item_impl(this, node)
            });
        }
    }
    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        if !test_only(&node.attrs) {
            self.scoped(format!("fn {}", node.sig.ident), |this| {
                syn::visit::visit_impl_item_fn(this, node)
            });
        }
    }
    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        if !test_only(&node.attrs) {
            self.scoped(format!("fn {}", node.sig.ident), |this| {
                syn::visit::visit_trait_item_fn(this, node)
            });
        }
    }
    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        if !test_only(&node.attrs) {
            self.scoped(format!("const {}", node.ident), |this| {
                syn::visit::visit_item_const(this, node)
            });
        }
    }
    fn visit_impl_item_const(&mut self, node: &'ast syn::ImplItemConst) {
        if !test_only(&node.attrs) {
            self.scoped(format!("const {}", node.ident), |this| {
                syn::visit::visit_impl_item_const(this, node)
            });
        }
    }
    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        if !test_only(&node.attrs) {
            self.scoped(format!("static {}", node.ident), |this| {
                syn::visit::visit_item_static(this, node)
            });
        }
    }
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if !test_only(&node.attrs) {
            syn::visit::visit_local(self, node);
        }
    }
    fn visit_arm(&mut self, node: &'ast syn::Arm) {
        if !test_only(&node.attrs) {
            syn::visit::visit_arm(self, node);
        }
    }
    fn visit_attribute(&mut self, node: &'ast Attribute) {
        match &node.meta {
            Meta::List(list)
                if !node.path().is_ident("cfg") && !node.path().is_ident("cfg_attr") =>
            {
                self.scan_tokens(list.tokens.clone());
            }
            _ => syn::visit::visit_attribute(self, node),
        }
    }
    fn visit_lit_str(&mut self, node: &'ast syn::LitStr) {
        let text = node.value();
        if is_lattice_name(&text) {
            self.record(&text, Kind::Literal);
        }
    }
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        match node.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
            Ok(arguments) => {
                for argument in &arguments {
                    self.visit_expr(argument);
                }
            }
            Err(_) => self.scan_tokens(node.tokens.clone()),
        }
    }
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let Expr::Path(function) = peel(&node.func)
            && last_segment(&function.path).as_deref() == Some("env_switch_enabled")
        {
            self.record_lookup(node, Kind::Switch);
        } else if is_env_lookup(node) {
            self.record_lookup(node, Kind::Value);
        }
        syn::visit::visit_expr_call(self, node);
    }
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if PRESENCE_METHODS.iter().any(|name| node.method == name) {
            let mut receiver = peel(&node.receiver);
            if let Expr::MethodCall(inner) = receiver
                && inner.method == "ok"
            {
                receiver = peel(&inner.receiver);
            }
            if let Some(call) = env_lookup_call(receiver) {
                self.record_lookup(call, Kind::Presence);
            }
        }
        syn::visit::visit_expr_method_call(self, node);
    }
    fn visit_expr_let(&mut self, node: &'ast syn::ExprLet) {
        if let syn::Pat::TupleStruct(pattern) = &*node.pat
            && matches!(last_segment(&pattern.path).as_deref(), Some("Some" | "Ok"))
            && pattern.elems.len() == 1
            && matches!(pattern.elems[0], syn::Pat::Wild(_))
            && let Some(call) = env_lookup_call(&node.expr)
        {
            self.record_lookup(call, Kind::Presence);
        }
        syn::visit::visit_expr_let(self, node);
    }
}

struct Population {
    hits: Vec<Hit>,
    non_test_files: usize,
    test_only_files: usize,
}

fn crate_roots(crate_dir: &Path) -> Vec<PathBuf> {
    let src = crate_dir.join("src");
    let mut roots = Vec::new();
    for name in ["lib.rs", "main.rs"] {
        if src.join(name).is_file() {
            roots.push(src.join(name));
        }
    }
    let bin = src.join("bin");
    if bin.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&bin)
            .expect("read src/bin")
            .map(|entry| entry.expect("read src/bin entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                if path.join("main.rs").is_file() {
                    roots.push(path.join("main.rs"));
                }
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                roots.push(path);
            }
        }
    }
    roots
}

fn closure(crate_dir: &Path, roots: &[PathBuf], test_cfg: bool) -> BTreeSet<PathBuf> {
    let mut files = BTreeSet::new();
    for root in roots {
        for source in module_source_closure(crate_dir, root, test_cfg)
            .unwrap_or_else(|reason| panic!("module closure for {}: {reason}", root.display()))
        {
            files.insert(source.path);
        }
    }
    files
}

fn scan_population_once() -> Population {
    let workspace = workspace_root();
    let crates_dir = workspace.join("crates");
    let mut crate_dirs: Vec<PathBuf> = std::fs::read_dir(&crates_dir)
        .expect("read crates directory")
        .map(|entry| entry.expect("read crates entry").path())
        .filter(|path| path.join("src").is_dir())
        .collect();
    crate_dirs.sort();

    // Parse each non-test file once and keep the trees, so the constant table is built
    // from the whole workspace before any lookup is resolved against it.
    let mut parsed: Vec<(String, String, syn::File)> = Vec::new();
    let mut non_test_files = 0;
    let mut test_only_files = 0;
    for crate_dir in &crate_dirs {
        let krate = crate_dir
            .file_name()
            .expect("crate directory name")
            .to_string_lossy()
            .into_owned();
        let roots = crate_roots(crate_dir);
        let canonical_crate = crate_dir.canonicalize().expect("resolve crate directory");
        let non_test = closure(&canonical_crate, &roots, false);
        let with_tests = closure(&canonical_crate, &roots, true);
        let every: BTreeSet<PathBuf> = rust_sources_under(&canonical_crate.join("src"))
            .into_iter()
            .map(|path| path.canonicalize().expect("resolve source path"))
            .collect();
        let reached: BTreeSet<PathBuf> = non_test.union(&with_tests).cloned().collect();
        let orphans: Vec<_> = every.difference(&reached).collect();
        assert!(
            orphans.is_empty(),
            "{krate}: source files reachable from no crate root in either cfg(test) setting, \
             so this guard cannot classify them: {orphans:?}"
        );
        non_test_files += non_test.len();
        test_only_files += with_tests.difference(&non_test).count();
        for path in non_test {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|reason| panic!("read {}: {reason}", path.display()));
            let tree = syn::parse_file(&text)
                .unwrap_or_else(|reason| panic!("parse {}: {reason}", path.display()));
            let relative = path
                .strip_prefix(&canonical_crate)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            parsed.push((krate.clone(), relative, tree));
        }
    }

    let mut collector = ConstCollector::default();
    for (_, _, tree) in &parsed {
        collector.visit_file(tree);
    }
    let mut hits = Vec::new();
    for (krate, file, tree) in &parsed {
        if test_only(&tree.attrs) {
            continue;
        }
        let mut scanner = Scanner {
            consts: &collector.values,
            krate,
            file,
            scope: Vec::new(),
            hits: Vec::new(),
        };
        scanner.visit_file(tree);
        hits.extend(scanner.hits);
    }
    Population {
        hits,
        non_test_files,
        test_only_files,
    }
}

/// Both tests read the same population; parsing ~400 files twice would double the cost.
fn scan_population() -> &'static Population {
    static POPULATION: std::sync::OnceLock<Population> = std::sync::OnceLock::new();
    POPULATION.get_or_init(scan_population_once)
}

fn config_rows() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("CONFIG.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|reason| panic!("read {}: {reason}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.trim_start().strip_prefix('|') else {
            continue;
        };
        let first = rest.split('|').next().unwrap_or("").trim();
        if let Some(name) = first
            .strip_prefix('`')
            .and_then(|inner| inner.strip_suffix('`'))
            && is_lattice_name(name)
        {
            rows.push(name.to_string());
        }
    }
    rows
}

fn names_of(hits: &[Hit]) -> BTreeSet<String> {
    hits.iter()
        .map(|hit| hit.name.clone())
        .filter(|name| name != UNRESOLVED_NAME)
        .collect()
}

/// Names a healthy scan must find. A scan that lost its discovery (an empty closure, a
/// parser that stopped matching) would otherwise find nothing and pass.
const ANCHORS: &[(&str, &str, Kind)] = &[
    ("LATTICE_NO_GPU", "inference", Kind::Switch),
    ("LATTICE_OFFLINE", "inference", Kind::Switch),
    ("LATTICE_MODEL_CACHE", "embed", Kind::Value),
    ("LATTICE_MODEL_CACHE", "inference", Kind::Value),
    ("LATTICE_MODEL_CACHE", "tune", Kind::Value),
    ("LATTICE_EMBED_DIM", "embed", Kind::Value),
];

fn assert_population(population: &Population) {
    let names = names_of(&population.hits);
    println!(
        "env_var_contract: {} non-test source files, {} test-only files, {} hits, {} distinct LATTICE_* names \
         ({} switch, {} value, {} presence, {} literal-only)",
        population.non_test_files,
        population.test_only_files,
        population.hits.len(),
        names.len(),
        names_with(population, Kind::Switch).len(),
        names_with(population, Kind::Value).len(),
        names_with(population, Kind::Presence).len(),
        names
            .iter()
            .filter(|name| {
                !population
                    .hits
                    .iter()
                    .any(|hit| &hit.name == *name && hit.kind != Kind::Literal)
            })
            .count(),
    );
    assert!(
        population.non_test_files >= 100,
        "found only {} non-test source files; discovery is broken",
        population.non_test_files
    );
    assert!(
        names.len() >= 30,
        "found only {} LATTICE_* names; discovery is broken",
        names.len()
    );
    for (name, krate, kind) in ANCHORS {
        assert!(
            population
                .hits
                .iter()
                .any(|hit| hit.name == *name && hit.krate == *krate && hit.kind == *kind),
            "anchor {name} ({kind:?}) not found in crate {krate}; discovery is broken"
        );
    }
}

fn names_with(population: &Population, kind: Kind) -> BTreeSet<String> {
    population
        .hits
        .iter()
        .filter(|hit| hit.kind == kind)
        .map(|hit| hit.name.clone())
        .collect()
}

#[test]
fn lattice_variables_are_not_read_by_presence() {
    let population = scan_population();
    assert_population(population);

    let allowed: BTreeSet<&str> = PRESENCE_ALLOWLIST.iter().map(|(name, _)| *name).collect();
    for (name, reason) in PRESENCE_ALLOWLIST {
        assert!(
            is_lattice_name(name) && !reason.is_empty(),
            "allowlist entry {name}"
        );
    }
    let offenders: Vec<String> = population
        .hits
        .iter()
        .filter(|hit| hit.kind == Kind::Presence && !allowed.contains(hit.name.as_str()))
        .map(|hit| {
            format!(
                "{} read by presence in crates/{}/{} ({})",
                hit.name, hit.krate, hit.file, hit.scope
            )
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "LATTICE_* variables read by presence (NAME=0 would enable them); read them through \
         `env_switch_enabled`, or allowlist the name with a reason:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn config_table_lists_exactly_the_variables_the_source_reads() {
    let population = scan_population();
    assert_population(population);

    let rows = config_rows();
    let row_set: BTreeSet<String> = rows.iter().cloned().collect();
    let source = names_of(&population.hits);
    println!(
        "env_var_contract: CONFIG.md has {} variable rows ({} distinct); source names {}",
        rows.len(),
        row_set.len(),
        source.len()
    );
    assert!(
        rows.len() >= 30,
        "parsed only {} CONFIG.md rows; table parse is broken",
        rows.len()
    );
    assert_eq!(
        rows.len(),
        row_set.len(),
        "CONFIG.md has a variable listed twice"
    );

    let missing: Vec<&String> = source.difference(&row_set).collect();
    let stale: Vec<&String> = row_set.difference(&source).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "crates/inference/CONFIG.md and the non-test source disagree.\n  read by source but no row: {missing:?}\n  \
         row but read by no non-test source: {stale:?}"
    );
}
