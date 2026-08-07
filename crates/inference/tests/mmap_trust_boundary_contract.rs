//! Contract test for #1368: every site in this crate that constructs a
//! memory map from a checkpoint file (`memmap2::Mmap::map(&file)` or
//! `memmap2::MmapOptions::new().map(&file)`) must call one of this crate's
//! two mmap trust-boundary chokepoints -- [`lattice_inference`]'s
//! `weights::mmap_trust::open_trusted_mmap_file` or
//! `reject_if_open_mmap_file_untrusted` -- **earlier in the same enclosing
//! function, on a path that necessarily executes before the construction
//! call** (see [`walk_expr`]'s doc comment for exactly what that does and
//! does not cover), or carry an explicit, reviewed
//! [`MmapConstructionExemption`] naming why not. A guard call that runs
//! after the construction call, or only on a sibling branch that does not
//! run before it, does not satisfy this.
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
use syn::{BinOp, Block, Expr, ExprCall, ExprMethodCall, ImplItem, Item, Stmt};

/// Free functions this crate's mmap trust boundary exposes. A call to either
/// one counts as guarding a construction site when [`walk_expr`] determines
/// it runs on a path that necessarily executes before the site, in the same
/// enclosing function -- not merely "anywhere in the function" (see
/// [`walk_expr`]'s doc comment for exactly what "necessarily executes before"
/// does and does not cover).
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
/// key format) and paired with its body for [`walk_block`]/[`walk_expr`]
/// below.
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

/// Whether an `Expr::Call` node calls one of [`GUARD_FUNCTION_NAMES`].
fn is_guard_call(call: &ExprCall) -> bool {
    matches!(&*call.func, Expr::Path(callee)
    if callee.path.segments.last().is_some_and(|segment| {
        GUARD_FUNCTION_NAMES.contains(&segment.ident.to_string().as_str())
    }))
}

/// Whether an `Expr::Call` node is the free-function `Mmap::map(..)` mmap
/// construction idiom.
fn is_mmap_map_call(call: &ExprCall) -> bool {
    matches!(&*call.func, Expr::Path(callee) if path_ends_with(&callee.path, &["Mmap", "map"]))
}

/// Whether an `Expr::MethodCall` node is the `MmapOptions::new().map(..)`
/// mmap construction idiom.
fn is_mmap_options_map_call(method_call: &ExprMethodCall) -> bool {
    method_call.method == "map"
        && matches!(&*method_call.receiver, Expr::Call(receiver_call)
            if receiver_call.args.is_empty()
                && matches!(&*receiver_call.func, Expr::Path(receiver_callee)
                    if path_ends_with(&receiver_callee.path, &["MmapOptions", "new"])))
}

/// A discovered mmap construction call, in the order [`walk_expr`] visits
/// it, paired with whether a guard call was seen on a path that necessarily
/// executes before it.
struct SiteObservation {
    selector: &'static str,
    dominated_by_guard: bool,
}

/// Walks an expression in source order, threading a `guarded` flag that
/// becomes (and stays) `true` once a guard call has been seen on a path that
/// *necessarily* executes before the current point -- the "first" the module
/// doc comment requires. This is a textual/structural approximation of
/// control-flow dominance, not a real control-flow-graph analysis; it is
/// sufficient for this crate's loaders, which are straight-line
/// open-then-map functions, but the following cases are explicitly NOT
/// treated as dominance:
///
/// - **`if`/`else` branches and `match` arms.** A guard call inside one arm
///   is scoped to that arm alone: it is never credited to a sibling arm, nor
///   to the statements that follow the conditional, because the *other* arm
///   might have run instead. This is what catches "one branch guards,
///   another maps unguarded". A conditional that guards on *every* arm is
///   (conservatively) still not credited outside the conditional -- doing
///   that soundly needs real control-flow reachability, which this AST walk
///   does not build.
/// - **Loop bodies** (`loop`/`while`/`for`). A loop may run zero times, so a
///   guard call inside one is never credited to code after the loop, and a
///   site inside the loop only sees guard calls from strictly before the
///   loop, never from an earlier iteration of the same loop.
/// - **Closures and `async`/`const` blocks.** Their bodies run later (or not
///   at all) relative to the enclosing function, so each starts its own,
///   independently threaded `guarded = false` and never reads or writes the
///   surrounding flow's flag in either direction.
/// - **Early `return`/`break`.** This walk has no notion of unreachable code
///   after a diverging expression: it keeps threading `guarded` through the
///   statements that textually follow one in the same block. A guard call
///   placed after an unconditional `return` would (incorrectly) still be
///   credited to sites after it, even though that guard call can never
///   actually run. None of this crate's loaders do that; it is recorded here
///   as a known gap rather than silently assumed away.
///
/// What DOES count as "necessarily executes before" and so threads the flag
/// straight through: sequential statements in the same block, `unsafe { .. }`
/// and bare `{ .. }` blocks, `?`, casts, references, field/index access,
/// method-call/call receivers and arguments (in left-to-right evaluation
/// order), assignment operands, and the operands of every binary operator
/// that Rust evaluates unconditionally (arithmetic, comparison, bitwise, ...).
///
/// **`&&`/`||` are handled as a branch boundary, not threaded through.** Like
/// `if`/`else`, Rust may skip the right operand entirely (`true || x` never
/// runs `x`; `false && x` never runs `x`), so a guard call inside the right
/// operand of `&&`/`||` is never credited to the binary expression's own
/// outgoing state, and so never reaches sites after it -- only whatever
/// `guarded` state the left operand (which always runs) produced does. The
/// right operand is still walked so a construction site inside it is
/// discovered and, symmetrically, still sees the left operand's guard state
/// as input, since the left operand necessarily runs first whenever the
/// right one runs at all.
fn walk_expr(expr: &Expr, guarded_in: bool, sites: &mut Vec<SiteObservation>) -> bool {
    match expr {
        Expr::Block(block) => walk_block(&block.block, guarded_in, sites),
        Expr::Unsafe(block) => walk_block(&block.block, guarded_in, sites),
        Expr::Call(call) => {
            let mut guarded = walk_expr(&call.func, guarded_in, sites);
            for arg in &call.args {
                guarded = walk_expr(arg, guarded, sites);
            }
            if is_mmap_map_call(call) {
                sites.push(SiteObservation {
                    selector: "Mmap::map()",
                    dominated_by_guard: guarded,
                });
            }
            if is_guard_call(call) {
                guarded = true;
            }
            guarded
        }
        Expr::MethodCall(method_call) => {
            let mut guarded = walk_expr(&method_call.receiver, guarded_in, sites);
            for arg in &method_call.args {
                guarded = walk_expr(arg, guarded, sites);
            }
            if is_mmap_options_map_call(method_call) {
                sites.push(SiteObservation {
                    selector: "MmapOptions::new().map()",
                    dominated_by_guard: guarded,
                });
            }
            guarded
        }
        Expr::If(if_expr) => {
            let guarded = walk_expr(&if_expr.cond, guarded_in, sites);
            walk_block(&if_expr.then_branch, guarded, sites);
            if let Some((_, else_expr)) = &if_expr.else_branch {
                walk_expr(else_expr, guarded, sites);
            }
            guarded
        }
        Expr::Match(match_expr) => {
            let guarded = walk_expr(&match_expr.expr, guarded_in, sites);
            for arm in &match_expr.arms {
                if let Some((_, guard_cond)) = &arm.guard {
                    walk_expr(guard_cond, guarded, sites);
                }
                walk_expr(&arm.body, guarded, sites);
            }
            guarded
        }
        Expr::Loop(loop_expr) => {
            walk_block(&loop_expr.body, guarded_in, sites);
            guarded_in
        }
        Expr::While(while_expr) => {
            let guarded = walk_expr(&while_expr.cond, guarded_in, sites);
            walk_block(&while_expr.body, guarded, sites);
            guarded_in
        }
        Expr::ForLoop(for_loop) => {
            let guarded = walk_expr(&for_loop.expr, guarded_in, sites);
            walk_block(&for_loop.body, guarded, sites);
            guarded_in
        }
        Expr::Closure(closure) => {
            walk_expr(&closure.body, false, sites);
            guarded_in
        }
        Expr::Try(try_expr) => walk_expr(&try_expr.expr, guarded_in, sites),
        Expr::Paren(paren) => walk_expr(&paren.expr, guarded_in, sites),
        Expr::Group(group) => walk_expr(&group.expr, guarded_in, sites),
        Expr::Reference(reference) => walk_expr(&reference.expr, guarded_in, sites),
        Expr::Unary(unary) => walk_expr(&unary.expr, guarded_in, sites),
        Expr::Cast(cast) => walk_expr(&cast.expr, guarded_in, sites),
        Expr::Field(field) => walk_expr(&field.base, guarded_in, sites),
        Expr::Await(await_expr) => walk_expr(&await_expr.base, guarded_in, sites),
        Expr::Let(let_expr) => walk_expr(&let_expr.expr, guarded_in, sites),
        Expr::Return(return_expr) => match &return_expr.expr {
            Some(inner) => walk_expr(inner, guarded_in, sites),
            None => guarded_in,
        },
        Expr::Break(break_expr) => match &break_expr.expr {
            Some(inner) => walk_expr(inner, guarded_in, sites),
            None => guarded_in,
        },
        Expr::Index(index) => {
            let guarded = walk_expr(&index.expr, guarded_in, sites);
            walk_expr(&index.index, guarded, sites)
        }
        Expr::Binary(binary) => {
            let guarded = walk_expr(&binary.left, guarded_in, sites);
            match binary.op {
                BinOp::And(_) | BinOp::Or(_) => {
                    // `&&`/`||` short-circuit: the right operand may never run.
                    // Walk it so a construction site inside is still discovered,
                    // but a guard call inside it must not be credited to sites
                    // that follow the whole expression -- only `guarded` (the
                    // state after the left operand, which always runs) does.
                    walk_expr(&binary.right, guarded, sites);
                    guarded
                }
                _ => walk_expr(&binary.right, guarded, sites),
            }
        }
        Expr::Assign(assign) => {
            let guarded = walk_expr(&assign.left, guarded_in, sites);
            walk_expr(&assign.right, guarded, sites)
        }
        Expr::Repeat(repeat) => {
            let guarded = walk_expr(&repeat.expr, guarded_in, sites);
            walk_expr(&repeat.len, guarded, sites)
        }
        Expr::Range(range) => {
            let mut guarded = guarded_in;
            if let Some(start) = &range.start {
                guarded = walk_expr(start, guarded, sites);
            }
            if let Some(end) = &range.end {
                guarded = walk_expr(end, guarded, sites);
            }
            guarded
        }
        Expr::Tuple(tuple) => {
            let mut guarded = guarded_in;
            for elem in &tuple.elems {
                guarded = walk_expr(elem, guarded, sites);
            }
            guarded
        }
        Expr::Array(array) => {
            let mut guarded = guarded_in;
            for elem in &array.elems {
                guarded = walk_expr(elem, guarded, sites);
            }
            guarded
        }
        Expr::Struct(struct_expr) => {
            let mut guarded = guarded_in;
            for field in &struct_expr.fields {
                guarded = walk_expr(&field.expr, guarded, sites);
            }
            if let Some(rest) = &struct_expr.rest {
                guarded = walk_expr(rest, guarded, sites);
            }
            guarded
        }
        // Every other expression kind (literals, paths, macros, `async`/
        // `const` blocks, ...) is either a leaf with nothing to recurse into,
        // or -- for `async`/`const` blocks -- runs later/independently like a
        // closure and is out of scope for the loaders this test discovers.
        _ => guarded_in,
    }
}

/// Walks a block's statements in source order, threading `guarded` the same
/// way [`walk_expr`] does.
fn walk_block(block: &Block, guarded_in: bool, sites: &mut Vec<SiteObservation>) -> bool {
    let mut guarded = guarded_in;
    for stmt in &block.stmts {
        guarded = match stmt {
            Stmt::Local(local) => {
                let mut guarded_after = guarded;
                if let Some(init) = &local.init {
                    guarded_after = walk_expr(&init.expr, guarded_after, sites);
                    if let Some((_, diverge)) = &init.diverge {
                        // `let ... else { diverge }`: the diverge block only
                        // runs when the pattern does NOT match, so -- like an
                        // `if`/`match` arm -- it is internal-only and never
                        // propagated to the statements that follow.
                        walk_expr(diverge, guarded_after, sites);
                    }
                }
                guarded_after
            }
            Stmt::Expr(expr, _) => walk_expr(expr, guarded, sites),
            Stmt::Macro(_) | Stmt::Item(_) => guarded,
        };
    }
    guarded
}

/// Regression fixture for the `||` short-circuit hole: `true || guard(..)`
/// never evaluates the guard at runtime, so it must not be credited to the
/// `Mmap::map()` call that follows. Parses a synthetic [`Block`] directly and
/// drives [`walk_block`] on it, rather than routing through the `src/` scan
/// and [`EXPECTED_MMAP_CONSTRUCTION_SITES`] inventory used by the tests
/// below -- this file has no existing indirection for exercising a shape
/// without a real construction site in `src/`, and adding a `src/`-only
/// fixture would touch production source for a case that has nothing to do
/// with a real loader.
#[test]
fn walk_expr_does_not_credit_a_guard_call_inside_an_or_short_circuit() {
    let block: Block = syn::parse_str(
        r#"{
            true || open_trusted_mmap_file(&file).is_ok();
            Mmap::map(&file)
        }"#,
    )
    .expect("parse `||` short-circuit fixture");
    let mut sites = Vec::new();
    walk_block(&block, false, &mut sites);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly the Mmap::map() construction site"
    );
    assert_eq!(sites[0].selector, "Mmap::map()");
    assert!(
        !sites[0].dominated_by_guard,
        "a guard call reachable only through the right operand of `||` must not be credited: \
         `true || open_trusted_mmap_file(..)` never evaluates the guard at runtime"
    );
}

/// Mirror of the `||` fixture above for `&&`: `false && guard(..)` never
/// evaluates the guard at runtime either.
#[test]
fn walk_expr_does_not_credit_a_guard_call_inside_an_and_short_circuit() {
    let block: Block = syn::parse_str(
        r#"{
            false && open_trusted_mmap_file(&file).is_ok();
            Mmap::map(&file)
        }"#,
    )
    .expect("parse `&&` short-circuit fixture");
    let mut sites = Vec::new();
    walk_block(&block, false, &mut sites);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly the Mmap::map() construction site"
    );
    assert_eq!(sites[0].selector, "Mmap::map()");
    assert!(
        !sites[0].dominated_by_guard,
        "a guard call reachable only through the right operand of `&&` must not be credited: \
         `false && open_trusted_mmap_file(..)` never evaluates the guard at runtime"
    );
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
            let mut sites = Vec::new();
            walk_block(body, false, &mut sites);
            if sites.is_empty() {
                continue;
            }

            let mut ordinals: BTreeMap<&str, usize> = BTreeMap::new();
            for site in &sites {
                let ordinal = ordinals.entry(site.selector).or_insert(0);
                *ordinal += 1;
                let selector = site.selector;
                let key = format!("{relative}::{function_path}::{selector}#{ordinal}");
                discovered.insert(key.clone());

                if !site.dominated_by_guard
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
        "mmap construction site(s) with neither a trust-boundary guard call earlier in the \
         same function on a path that necessarily reaches the site, nor a reviewed \
         MmapConstructionExemption:\n{}",
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
