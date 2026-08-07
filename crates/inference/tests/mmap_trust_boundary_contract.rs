//! Contract test for #1368: every site in this crate that constructs a
//! memory map from a checkpoint file (`memmap2::Mmap::map(&file)` or
//! `memmap2::MmapOptions::new().map(&file)`) must first call one of this
//! crate's two mmap trust-boundary chokepoints --
//! [`lattice_inference`]'s `weights::mmap_trust::open_trusted_mmap_file` or
//! `reject_if_open_mmap_file_untrusted` -- in the same enclosing function, or
//! carry an explicit, reviewed [`MmapConstructionExemption`] naming why not.
//!
//! Modeled on `metal_measurement_lock_contract.rs`'s construction-site
//! inventory: a site is named by the program structure enclosing it
//! (`mod`/`impl`/`fn` path, plus an ordinal disambiguating repeated calls in
//! the same function) rather than by source line/column, so an edit that
//! only shifts line numbers cannot silently invalidate an exemption or
//! duplicate an entry -- a position-keyed guard has to be repaired by
//! exactly the motion that would silence it. The full discovered inventory
//! (guarded sites plus exempted sites) is asserted against a declared
//! expected set so an added or removed construction site always requires a
//! conscious update here, whether or not it happens to be guarded.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprMethodCall, ImplItem, Item};

/// Free functions this crate's mmap trust boundary exposes. A call to either
/// one, anywhere in the function enclosing a construction site, counts as
/// that site being guarded.
const GUARD_FUNCTION_NAMES: &[&str] = &[
    "open_trusted_mmap_file",
    "reject_if_open_mmap_file_untrusted",
];

/// A construction site exempted from the guard-call requirement, keyed by
/// program structure (see module doc comment), with a reviewed reason on
/// record for why routing it through a guard call directly is unnecessary
/// or incorrect.
struct MmapConstructionExemption {
    site: &'static str,
    reason: &'static str,
}

const MMAP_CONSTRUCTION_EXEMPTIONS: &[MmapConstructionExemption] = &[MmapConstructionExemption {
    site: "src/weights/f32_weights.rs::SafetensorsFile::from_open_file::Mmap::map()#1",
    reason: "from_open_file takes an already-open File specifically so it never re-opens \
             by path (reopening would reintroduce the window between open and read its own \
             doc comment describes). Its only callers -- SafetensorsFile::open and \
             ShardedSafetensors::open_shard (via open_manifest_entry_once) -- both call \
             reject_if_open_mmap_file_untrusted on that same File before handing it here, \
             so the guard call sits in the caller, not in this function.",
}];

/// The complete population of mmap-construction sites this crate contains,
/// whether guarded directly or exempted above. Discovering a site not in
/// this set, or failing to discover one that is, fails the contract --
/// forcing a conscious update here for any added, removed, or newly
/// (un)guarded site.
const EXPECTED_MMAP_CONSTRUCTION_SITES: &[&str] = &[
    "src/forward/metal_qwen35.rs::inner::mmap_q3_weight::MmapOptions::new().map()#1",
    "src/quant/quarot/io.rs::Shard::open::Mmap::map()#1",
    "src/weights/f32_weights.rs::SafetensorsFile::from_open_file::Mmap::map()#1",
    "src/weights/mmap_trust.rs::tests::verify_mmap_target_unchanged_accepts_an_unmodified_file::MmapOptions::new().map()#1",
    "src/weights/mmap_trust.rs::tests::verify_mmap_target_unchanged_rejects_a_truncate_after_validate_race::MmapOptions::new().map()#1",
    "src/weights/q4_weights.rs::open_and_mmap_q4_file::MmapOptions::new().map()#1",
];

fn rust_sources_under(root: &Path) -> Vec<PathBuf> {
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

fn path_ends_with(path: &syn::Path, tail: &[&str]) -> bool {
    let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    if segments.len() < tail.len() {
        return false;
    }
    segments[segments.len() - tail.len()..]
        .iter()
        .zip(tail.iter())
        .all(|(have, want)| have == want)
}

/// The receiver type name for an `impl` block, mirroring the sibling Metal
/// contract test's `impl_self_type_label` so a method's stable key includes
/// its type even when two `impl` blocks define a method with the same name.
fn impl_self_type_label(self_ty: &syn::Type) -> String {
    if let syn::Type::Path(type_path) = self_ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident.to_string();
    }
    "impl".to_string()
}

/// Every free function and `impl` method in a parsed file, named by its
/// enclosing `mod`/`impl` path (matching `EXPECTED_MMAP_CONSTRUCTION_SITES`'
/// key format) and paired with its body for the two finder visitors below.
fn collect_functions<'a>(
    items: &'a [Item],
    path: &mut Vec<String>,
    out: &mut Vec<(String, &'a syn::Block)>,
) {
    for item in items {
        match item {
            Item::Fn(function) => {
                path.push(function.sig.ident.to_string());
                out.push((path.join("::"), &function.block));
                path.pop();
            }
            Item::Mod(module) => {
                if let Some((_, contents)) = &module.content {
                    path.push(module.ident.to_string());
                    collect_functions(contents, path, out);
                    path.pop();
                }
            }
            Item::Impl(item_impl) => {
                path.push(impl_self_type_label(&item_impl.self_ty));
                for impl_item in &item_impl.items {
                    if let ImplItem::Fn(method) = impl_item {
                        path.push(method.sig.ident.to_string());
                        out.push((path.join("::"), &method.block));
                        path.pop();
                    }
                }
                path.pop();
            }
            _ => {}
        }
    }
}

/// Finds every `Mmap::map(..)` / `MmapOptions::new().map(..)` construction
/// call in a function body, in visitation order, labeled by which of the two
/// idioms matched.
#[derive(Default)]
struct MmapConstructionFinder {
    sites: Vec<&'static str>,
}

impl<'ast> Visit<'ast> for MmapConstructionFinder {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(callee) = &*node.func
            && path_ends_with(&callee.path, &["Mmap", "map"])
        {
            self.sites.push("Mmap::map()");
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        if node.method == "map"
            && let Expr::Call(receiver_call) = &*node.receiver
            && receiver_call.args.is_empty()
            && let Expr::Path(receiver_callee) = &*receiver_call.func
            && path_ends_with(&receiver_callee.path, &["MmapOptions", "new"])
        {
            self.sites.push("MmapOptions::new().map()");
        }
        visit::visit_expr_method_call(self, node);
    }
}

/// Whether a function body calls one of `GUARD_FUNCTION_NAMES`, anywhere in
/// its body (including nested blocks/closures) -- these loaders are
/// straight-line open-then-mmap functions, so "called somewhere in this
/// function" is the property the mutation-arm test in #1368's PR actually
/// exercises, not a stricter before/after ordering.
#[derive(Default)]
struct GuardCallFinder {
    found: bool,
}

impl<'ast> Visit<'ast> for GuardCallFinder {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(callee) = &*node.func
            && let Some(last) = callee.path.segments.last()
            && GUARD_FUNCTION_NAMES.contains(&last.ident.to_string().as_str())
        {
            self.found = true;
        }
        visit::visit_expr_call(self, node);
    }
}

#[test]
fn every_mmap_construction_site_is_guarded_or_explicitly_exempted() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_root = manifest_dir.join("src");

    let mut discovered: BTreeSet<String> = BTreeSet::new();
    let mut violations: Vec<String> = Vec::new();

    for path in rust_sources_under(&src_root) {
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|reason| panic!("could not read {relative}: {reason}"));
        let syntax = syn::parse_file(&source).unwrap_or_else(|reason| {
            panic!("{relative}: Rust syntax could not be parsed: {reason}")
        });

        let mut functions = Vec::new();
        let mut path_stack = Vec::new();
        collect_functions(&syntax.items, &mut path_stack, &mut functions);

        for (function_path, body) in functions {
            let mut construction = MmapConstructionFinder::default();
            construction.visit_block(body);
            if construction.sites.is_empty() {
                continue;
            }

            let mut guard = GuardCallFinder::default();
            guard.visit_block(body);

            let mut ordinals: BTreeMap<&str, usize> = BTreeMap::new();
            for selector in &construction.sites {
                let ordinal = ordinals.entry(selector).or_insert(0);
                *ordinal += 1;
                let key = format!("{relative}::{function_path}::{selector}#{ordinal}");
                discovered.insert(key.clone());

                if !guard.found
                    && !MMAP_CONSTRUCTION_EXEMPTIONS
                        .iter()
                        .any(|exemption| exemption.site == key)
                {
                    violations.push(key);
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "mmap construction site(s) with neither a trust-boundary guard call in the same \
         function nor a reviewed MmapConstructionExemption:\n{}",
        violations.join("\n")
    );

    let expected: BTreeSet<String> = EXPECTED_MMAP_CONSTRUCTION_SITES
        .iter()
        .map(|site| (*site).to_string())
        .collect();
    assert_eq!(
        discovered, expected,
        "mmap construction-site inventory changed; classify every added or removed site \
         explicitly -- guard it with open_trusted_mmap_file / \
         reject_if_open_mmap_file_untrusted, or add a reviewed MmapConstructionExemption \
         naming why not, then update EXPECTED_MMAP_CONSTRUCTION_SITES"
    );
}

/// Every exemption entry must actually correspond to a discovered
/// construction site: a stale entry (the site was fixed, renamed, or
/// deleted) would silently stop meaning anything.
#[test]
fn every_exemption_matches_a_currently_expected_site() {
    let expected: BTreeSet<&str> = EXPECTED_MMAP_CONSTRUCTION_SITES.iter().copied().collect();
    for exemption in MMAP_CONSTRUCTION_EXEMPTIONS {
        assert!(
            expected.contains(exemption.site),
            "exemption `{}` does not match any site in EXPECTED_MMAP_CONSTRUCTION_SITES -- \
             it is stale and should be removed",
            exemption.site
        );
        assert!(
            !exemption.reason.is_empty(),
            "exemption `{}` must state a reason",
            exemption.site
        );
    }
}
