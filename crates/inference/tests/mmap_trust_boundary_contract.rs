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
//! run before it, does not satisfy this. Nor does *visiting* a guard call
//! textually: the guard returns a `Result`, and only a form that actually
//! propagates its failure -- `?`, `.expect(..)`, `.unwrap()` -- credits the
//! statements after it as guarded. Collapsing that `Result` to a `bool`
//! first (`.is_ok()`, `.is_err()`) and branching on the bool does not, even
//! though the raw guard call is still textually present somewhere in the
//! condition (see [`is_guard_result_chain`]).
//!
//! A function whose body contains an opaque form this walk cannot see into
//! (a macro invocation, or a `Verbatim` fragment syn itself could not
//! classify) is also a violation **if that function contains at least one
//! discovered construction site** -- the opaque form might expand to hide
//! another one, or a guard call this walk would otherwise have credited,
//! and there is no way to tell from the AST alone. A function with no
//! construction site has nothing for an opaque form to hide, so it is not
//! flagged; without that qualifier, this crate's ordinary use of `assert!`/
//! `format!`/`vec!` outside closures anywhere in `src/` would fail every
//! function that happens to use one, not just loaders (see
//! [`every_mmap_construction_site_is_guarded_or_explicitly_exempted`]).
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
use syn::{BinOp, Block, Expr, ExprCall, ExprMethodCall, ImplItem, Item, Stmt, TraitItem};

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

/// Macros this walk trusts as pure, control-flow-transparent value
/// producers with no bearing on the mmap trust boundary -- all from `std`,
/// none able to expand into a hidden item, `let`, or arbitrary user call.
/// This crate's real guarded loaders use several of these directly (not
/// only inside a closure) for error-message construction, e.g.
/// `return Err(InferenceError::InvalidSafetensors(format!(".."))))` in
/// `quant/quarot/io.rs::Shard::open`, and `assert!`/`debug_assert!` as bare
/// validation statements in `weights/q4_weights.rs::open_and_mmap_q4_file`
/// and `weights/mmap_trust.rs`'s own tests. Any macro NOT on this list is
/// opaque (see [`walk_expr`]'s `Expr::Macro` arm and the module doc
/// comment): this is a reviewed, deliberate scope decision, not an
/// oversight -- treating literally every macro invocation in the crate as
/// disqualifying would fail functions that have nothing to do with mmap
/// construction at all (see
/// [`every_mmap_construction_site_is_guarded_or_explicitly_exempted`]'s doc
/// comment), which is a different failure mode than the one this test
/// exists to catch.
const TRANSPARENT_MACRO_NAMES: &[&str] = &[
    "format",
    "write",
    "writeln",
    "vec",
    "matches",
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
    "panic",
    "unreachable",
    "todo",
    "unimplemented",
    "println",
    "eprintln",
    "print",
    "eprint",
    "dbg",
];

/// Whether a macro path names one of [`TRANSPARENT_MACRO_NAMES`]. Matches
/// only a bare single-segment path (`format!(..)`), consistent with how all
/// of these are actually invoked -- a qualified path to a same-named macro
/// (`some_module::format!(..)`, vanishingly rare and not used anywhere in
/// this crate) is deliberately NOT trusted, the same "don't trust a bare
/// name across a module boundary" posture [`is_guard_call`] takes for the
/// opposite reason (there, to avoid crediting a decoy; here, to avoid
/// trusting one).
fn is_transparent_macro(path: &syn::Path) -> bool {
    path.get_ident()
        .is_some_and(|ident| TRANSPARENT_MACRO_NAMES.contains(&ident.to_string().as_str()))
}

/// The module (relative to the manifest directory) that defines
/// [`GUARD_FUNCTION_NAMES`]. Every caller outside this file spells the guard
/// call with the full crate-qualified path (`crate::weights::mmap_trust::..`
/// or `weights::mmap_trust::..`); only this module's own `mod tests` (via
/// `use super::*`) calls a guard bare. A bare, unqualified call anywhere
/// else in the crate cannot be trusted to name the real guard rather than a
/// same-named local shadow (see [`is_guard_call`]).
const GUARD_DEFINITION_MODULE: &str = "src/weights/mmap_trust.rs";

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

/// Renders a `syn::Path` as `a::b::c`, for the opaque-form messages this
/// walk records.
fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
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

/// Every free function, `impl` method, and default `trait` method body in a
/// parsed file, named by its enclosing `mod`/`impl`/`trait` path (matching
/// `EXPECTED_MMAP_CONSTRUCTION_SITES`' key format) and paired with its body
/// for [`walk_block`]/[`walk_expr`] below. Also descends one level into each
/// collected body's own top-level statements looking for a nested `fn`
/// item, so `fn outer() { fn inner() { .. } .. }` reaches `inner` as its own
/// entry rather than vanishing into `Stmt::Item`'s pass-through in
/// [`walk_block`]. A nested item declared inside an `if`/`match`/loop body
/// rather than directly in the function's own block is a further,
/// documented gap: reaching it needs the same block-walking [`walk_expr`]
/// already does, which this discovery pass deliberately keeps separate from
/// (rather than duplicating).
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
                collect_nested_function_items(&function.block, path, out);
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
                        collect_nested_function_items(&method.block, path, out);
                        path.pop();
                    }
                }
                path.pop();
            }
            Item::Trait(item_trait) => {
                path.push(item_trait.ident.to_string());
                for trait_item in &item_trait.items {
                    if let TraitItem::Fn(method) = trait_item
                        && let Some(default) = &method.default
                    {
                        path.push(method.sig.ident.to_string());
                        out.push((path.join("::"), default));
                        collect_nested_function_items(default, path, out);
                        path.pop();
                    }
                }
                path.pop();
            }
            _ => {}
        }
    }
}

/// Scans a block's own top-level statements (not nested blocks -- see
/// [`collect_functions`]'s doc comment) for a nested `fn`/`impl`/`trait`
/// item and feeds it back through [`collect_functions`] under the enclosing
/// path, so it gets its own inventory entry instead of silently vanishing.
fn collect_nested_function_items<'a>(
    block: &'a syn::Block,
    path: &mut Vec<String>,
    out: &mut Vec<(String, &'a syn::Block)>,
) {
    for stmt in &block.stmts {
        if let Stmt::Item(item) = stmt {
            collect_functions(std::slice::from_ref(item), path, out);
        }
    }
}

/// Whether an `Expr::Call` node calls one of [`GUARD_FUNCTION_NAMES`],
/// resolved through the crate-qualified path rather than the final path
/// segment alone. A bare, single-segment call is only accepted when scanning
/// [`GUARD_DEFINITION_MODULE`] itself (see that constant's doc comment) --
/// everywhere else a bare identifier could name a same-named local shadow,
/// and a multi-segment path that isn't the real `weights::mmap_trust::..`
/// suffix definitely does.
fn is_guard_call(call: &ExprCall, ctx: &WalkCtx) -> bool {
    let Expr::Path(callee) = &*call.func else {
        return false;
    };
    let segments: Vec<String> = callee
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    let Some(name) = segments.last() else {
        return false;
    };
    if !GUARD_FUNCTION_NAMES.contains(&name.as_str()) {
        return false;
    }
    match segments.len() {
        1 => ctx.in_guard_definition_module,
        _ => {
            let tail: &[String] = if segments[0] == "crate" {
                &segments[1..]
            } else {
                &segments[..]
            };
            tail.len() == 3 && tail[0] == "weights" && tail[1] == "mmap_trust"
        }
    }
}

/// Whether `expr` is a `Result`-valued chain rooted at a guard call and
/// carried through zero or more combinators that preserve its `Ok`-ness
/// (currently just `.map_err(..)`, the only one this crate's callers use).
/// Deliberately does NOT recognize `.is_ok()`/`.is_err()` or any other
/// combinator that collapses the `Result` to something else: those launder
/// the guard's outcome into a value whose truthiness this walk cannot
/// soundly connect back to "the guard succeeded" (see [`walk_expr`]'s
/// `Expr::Try`/`Expr::MethodCall` handling, which are the only places this
/// predicate feeds into the `guarded` flag).
fn is_guard_result_chain(expr: &Expr, ctx: &WalkCtx) -> bool {
    match expr {
        Expr::Call(call) => is_guard_call(call, ctx),
        Expr::MethodCall(method_call) if method_call.method == "map_err" => {
            is_guard_result_chain(&method_call.receiver, ctx)
        }
        Expr::Paren(paren) => is_guard_result_chain(&paren.expr, ctx),
        Expr::Group(group) => is_guard_result_chain(&group.expr, ctx),
        _ => false,
    }
}

/// Whether an `Expr::Call` node is the inline `MmapOptions::new()` idiom.
fn is_mmap_options_new_call(call: &ExprCall) -> bool {
    call.args.is_empty()
        && matches!(&*call.func, Expr::Path(callee) if path_ends_with(&callee.path, &["MmapOptions", "new"]))
}

/// Whether an `Expr::Call` node is the free-function `Mmap::map(..)` mmap
/// construction idiom. Syntactic, like [`is_mmap_options_new_call`]: an
/// alias or function-item indirection of `Mmap::map` (`let f = Mmap::map; f
/// (&file)`) is not recognized, and is a documented residual gap rather than
/// a silently assumed-away one -- no construction site in this crate uses
/// that shape today.
fn is_mmap_map_call(call: &ExprCall) -> bool {
    matches!(&*call.func, Expr::Path(callee) if path_ends_with(&callee.path, &["Mmap", "map"]))
}

/// Whether an `Expr::MethodCall` node is the `MmapOptions::new().map(..)`
/// mmap construction idiom, including when the `MmapOptions::new()` value
/// was bound to a local variable first (`let opts = MmapOptions::new();
/// opts.map(..)`) -- tracked via [`WalkCtx::bound_mmap_options`], populated
/// by [`walk_block`] as it processes `let` statements.
fn is_mmap_options_map_call(method_call: &ExprMethodCall, ctx: &WalkCtx) -> bool {
    if method_call.method != "map" {
        return false;
    }
    match &*method_call.receiver {
        Expr::Call(receiver_call) => is_mmap_options_new_call(receiver_call),
        Expr::Path(receiver_path) => receiver_path
            .path
            .get_ident()
            .is_some_and(|ident| ctx.bound_mmap_options.contains(&ident.to_string())),
        _ => false,
    }
}

/// A discovered mmap construction call, in the order [`walk_expr`] visits
/// it, paired with whether a guard call was seen on a path that necessarily
/// executes before it.
struct SiteObservation {
    selector: &'static str,
    dominated_by_guard: bool,
}

/// Context threaded through [`walk_expr`]/[`walk_block`] alongside the
/// per-branch `guarded` flag: state that is scoped to (part of) a function
/// rather than to one control-flow branch.
#[derive(Clone)]
struct WalkCtx<'a> {
    /// `{relative_path}::{function_path}`, named in an opaque-form message
    /// so it points at the function, not just "somewhere".
    location: &'a str,
    /// See [`GUARD_DEFINITION_MODULE`].
    in_guard_definition_module: bool,
    /// True once inside a closure/`async`/`const` block. These run later,
    /// independently of the enclosing function -- see the "Closures and
    /// `async`/`const` blocks" bullet in [`walk_expr`]'s doc comment -- so
    /// an opaque macro/`Verbatim` form inside one cannot hide code that runs
    /// on *this* function's own guard-to-map path, and is not recorded (see
    /// the `Expr::Macro`/`Expr::Verbatim` arms below).
    in_deferred_scope: bool,
    /// Local variables in the current scope bound directly to
    /// `MmapOptions::new()` (see [`is_mmap_options_map_call`]).
    bound_mmap_options: BTreeSet<String>,
}

fn empty_ctx(location: &str) -> WalkCtx<'_> {
    WalkCtx {
        location,
        in_guard_definition_module: false,
        in_deferred_scope: false,
        bound_mmap_options: BTreeSet::new(),
    }
}

/// Walks an expression in source order, threading a `guarded` flag that
/// becomes (and stays) `true` once a guard call has been seen on a path that
/// *necessarily* executes before the current point -- the "first" the module
/// doc comment requires -- AND whose success is actually propagated, not
/// merely observed (see [`is_guard_result_chain`]). This is a
/// textual/structural approximation of control-flow dominance, not a real
/// control-flow-graph analysis; it is sufficient for this crate's loaders,
/// which are straight-line open-then-map functions, but the following cases
/// are explicitly NOT treated as dominance:
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
/// - **A guard call collapsed to a `bool`.** `guard(..).is_ok()` or
///   `guard(..).is_err()`, used as an `if`/`&&`/`||` operand, does NOT credit
///   the guard: visiting the call is not the same as observing that it
///   succeeded and having that success actually gate what runs next. Only
///   `guard(..)?`, `guard(..).expect(..)`, `guard(..).unwrap()` (optionally
///   through `.map_err(..)`) do that -- see [`is_guard_result_chain`] and the
///   `Expr::Try`/`Expr::MethodCall` arms below.
///
/// What DOES count as "necessarily executes before" and so threads the flag
/// straight through: sequential statements in the same block, `unsafe { .. }`
/// and bare `{ .. }` blocks, `try { .. }` blocks, `?`, casts, references,
/// raw address-of (`&raw const`/`&raw mut`), field/index access,
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
///
/// **Every `Expr` variant syn 2.0.119 defines is matched explicitly.** A
/// pure leaf with nothing to recurse into (`Continue`, `Infer`, `Lit`,
/// `Path`) passes `guarded_in` straight through. An opaque form this walk
/// cannot see into -- `Macro`, `Verbatim` -- is recorded into `opaque`
/// (unless `ctx.in_deferred_scope`, see [`WalkCtx::in_deferred_scope`]);
/// [`every_mmap_construction_site_is_guarded_or_explicitly_exempted`] turns
/// that into a violation for any function that also has a discovered
/// construction site. `Expr` is `#[non_exhaustive]`, so a `_` arm is still
/// required for a future syn version's new variant; unlike the two known
/// opaque forms, that arm panics immediately -- a variant this walk has
/// never seen is not something the deferred, per-function judgment above
/// can safely reason about at all.
fn walk_expr(
    expr: &Expr,
    guarded_in: bool,
    ctx: &WalkCtx,
    sites: &mut Vec<SiteObservation>,
    opaque: &mut Vec<String>,
) -> bool {
    match expr {
        Expr::Block(block) => walk_block(&block.block, guarded_in, ctx, sites, opaque),
        Expr::Unsafe(block) => walk_block(&block.block, guarded_in, ctx, sites, opaque),
        Expr::TryBlock(try_block) => walk_block(&try_block.block, guarded_in, ctx, sites, opaque),
        Expr::Call(call) => {
            let mut guarded = walk_expr(&call.func, guarded_in, ctx, sites, opaque);
            for arg in &call.args {
                guarded = walk_expr(arg, guarded, ctx, sites, opaque);
            }
            if is_mmap_map_call(call) {
                sites.push(SiteObservation {
                    selector: "Mmap::map()",
                    dominated_by_guard: guarded,
                });
            }
            guarded
        }
        Expr::MethodCall(method_call) => {
            let mut guarded = walk_expr(&method_call.receiver, guarded_in, ctx, sites, opaque);
            for arg in &method_call.args {
                guarded = walk_expr(arg, guarded, ctx, sites, opaque);
            }
            if is_mmap_options_map_call(method_call, ctx) {
                sites.push(SiteObservation {
                    selector: "MmapOptions::new().map()",
                    dominated_by_guard: guarded,
                });
            }
            if (method_call.method == "expect" || method_call.method == "unwrap")
                && is_guard_result_chain(&method_call.receiver, ctx)
            {
                guarded = true;
            }
            guarded
        }
        Expr::Try(try_expr) => {
            let guarded = walk_expr(&try_expr.expr, guarded_in, ctx, sites, opaque);
            if is_guard_result_chain(&try_expr.expr, ctx) {
                true
            } else {
                guarded
            }
        }
        Expr::If(if_expr) => {
            let guarded = walk_expr(&if_expr.cond, guarded_in, ctx, sites, opaque);
            walk_block(&if_expr.then_branch, guarded, ctx, sites, opaque);
            if let Some((_, else_expr)) = &if_expr.else_branch {
                walk_expr(else_expr, guarded, ctx, sites, opaque);
            }
            guarded
        }
        Expr::Match(match_expr) => {
            let guarded = walk_expr(&match_expr.expr, guarded_in, ctx, sites, opaque);
            for arm in &match_expr.arms {
                if let Some((_, guard_cond)) = &arm.guard {
                    walk_expr(guard_cond, guarded, ctx, sites, opaque);
                }
                walk_expr(&arm.body, guarded, ctx, sites, opaque);
            }
            guarded
        }
        Expr::Loop(loop_expr) => {
            walk_block(&loop_expr.body, guarded_in, ctx, sites, opaque);
            guarded_in
        }
        Expr::While(while_expr) => {
            let guarded = walk_expr(&while_expr.cond, guarded_in, ctx, sites, opaque);
            walk_block(&while_expr.body, guarded, ctx, sites, opaque);
            guarded_in
        }
        Expr::ForLoop(for_loop) => {
            let guarded = walk_expr(&for_loop.expr, guarded_in, ctx, sites, opaque);
            walk_block(&for_loop.body, guarded, ctx, sites, opaque);
            guarded_in
        }
        Expr::Closure(closure) => {
            let mut deferred = ctx.clone();
            deferred.in_deferred_scope = true;
            walk_expr(&closure.body, false, &deferred, sites, opaque);
            guarded_in
        }
        Expr::Async(async_expr) => {
            let mut deferred = ctx.clone();
            deferred.in_deferred_scope = true;
            walk_block(&async_expr.block, false, &deferred, sites, opaque);
            guarded_in
        }
        Expr::Const(const_expr) => {
            let mut deferred = ctx.clone();
            deferred.in_deferred_scope = true;
            walk_block(&const_expr.block, false, &deferred, sites, opaque);
            guarded_in
        }
        Expr::Paren(paren) => walk_expr(&paren.expr, guarded_in, ctx, sites, opaque),
        Expr::Group(group) => walk_expr(&group.expr, guarded_in, ctx, sites, opaque),
        Expr::Reference(reference) => walk_expr(&reference.expr, guarded_in, ctx, sites, opaque),
        Expr::RawAddr(raw_addr) => walk_expr(&raw_addr.expr, guarded_in, ctx, sites, opaque),
        Expr::Unary(unary) => walk_expr(&unary.expr, guarded_in, ctx, sites, opaque),
        Expr::Cast(cast) => walk_expr(&cast.expr, guarded_in, ctx, sites, opaque),
        Expr::Field(field) => walk_expr(&field.base, guarded_in, ctx, sites, opaque),
        Expr::Await(await_expr) => walk_expr(&await_expr.base, guarded_in, ctx, sites, opaque),
        Expr::Let(let_expr) => walk_expr(&let_expr.expr, guarded_in, ctx, sites, opaque),
        Expr::Return(return_expr) => match &return_expr.expr {
            Some(inner) => walk_expr(inner, guarded_in, ctx, sites, opaque),
            None => guarded_in,
        },
        Expr::Break(break_expr) => match &break_expr.expr {
            Some(inner) => walk_expr(inner, guarded_in, ctx, sites, opaque),
            None => guarded_in,
        },
        Expr::Yield(yield_expr) => match &yield_expr.expr {
            Some(inner) => walk_expr(inner, guarded_in, ctx, sites, opaque),
            None => guarded_in,
        },
        Expr::Continue(_) | Expr::Infer(_) | Expr::Lit(_) | Expr::Path(_) => guarded_in,
        Expr::Index(index) => {
            let guarded = walk_expr(&index.expr, guarded_in, ctx, sites, opaque);
            walk_expr(&index.index, guarded, ctx, sites, opaque)
        }
        Expr::Binary(binary) => {
            let guarded = walk_expr(&binary.left, guarded_in, ctx, sites, opaque);
            match binary.op {
                BinOp::And(_) | BinOp::Or(_) => {
                    // `&&`/`||` short-circuit: the right operand may never run.
                    // Walk it so a construction site inside is still discovered,
                    // but a guard call inside it must not be credited to sites
                    // that follow the whole expression -- only `guarded` (the
                    // state after the left operand, which always runs) does.
                    walk_expr(&binary.right, guarded, ctx, sites, opaque);
                    guarded
                }
                _ => walk_expr(&binary.right, guarded, ctx, sites, opaque),
            }
        }
        Expr::Assign(assign) => {
            let guarded = walk_expr(&assign.left, guarded_in, ctx, sites, opaque);
            walk_expr(&assign.right, guarded, ctx, sites, opaque)
        }
        Expr::Repeat(repeat) => {
            let guarded = walk_expr(&repeat.expr, guarded_in, ctx, sites, opaque);
            walk_expr(&repeat.len, guarded, ctx, sites, opaque)
        }
        Expr::Range(range) => {
            let mut guarded = guarded_in;
            if let Some(start) = &range.start {
                guarded = walk_expr(start, guarded, ctx, sites, opaque);
            }
            if let Some(end) = &range.end {
                guarded = walk_expr(end, guarded, ctx, sites, opaque);
            }
            guarded
        }
        Expr::Tuple(tuple) => {
            let mut guarded = guarded_in;
            for elem in &tuple.elems {
                guarded = walk_expr(elem, guarded, ctx, sites, opaque);
            }
            guarded
        }
        Expr::Array(array) => {
            let mut guarded = guarded_in;
            for elem in &array.elems {
                guarded = walk_expr(elem, guarded, ctx, sites, opaque);
            }
            guarded
        }
        Expr::Struct(struct_expr) => {
            let mut guarded = guarded_in;
            for field in &struct_expr.fields {
                guarded = walk_expr(&field.expr, guarded, ctx, sites, opaque);
            }
            if let Some(rest) = &struct_expr.rest {
                guarded = walk_expr(rest, guarded, ctx, sites, opaque);
            }
            guarded
        }
        Expr::Macro(macro_expr) => {
            if !ctx.in_deferred_scope && !is_transparent_macro(&macro_expr.mac.path) {
                opaque.push(format!(
                    "{}: a macro invocation (`{}!(..)`) is opaque to this AST walk",
                    ctx.location,
                    path_to_string(&macro_expr.mac.path)
                ));
            }
            guarded_in
        }
        Expr::Verbatim(_) => {
            if !ctx.in_deferred_scope {
                opaque.push(format!(
                    "{}: syn could not classify this expression into a known AST form \
                     (Verbatim)",
                    ctx.location
                ));
            }
            guarded_in
        }
        // `Expr` is `#[non_exhaustive]`: every variant syn 2.0.119 defines is
        // matched explicitly above (see this function's doc comment). A
        // variant this match doesn't recognize -- from a newer syn version --
        // fails closed immediately rather than being folded into the
        // deferred `opaque` judgment above: upgrading syn to a version with
        // a genuinely new `Expr` variant needs a conscious update here, not
        // a per-function pass/fail call this walk cannot make blind.
        _ => panic!(
            "{}: an expression form this AST walk does not recognize (syn may have added a new \
             `Expr` variant) -- update `walk_expr` to handle it explicitly before trusting this \
             contract again",
            ctx.location
        ),
    }
}

/// Walks a block's statements in source order, threading `guarded` the same
/// way [`walk_expr`] does, and extending `ctx.bound_mmap_options` as `let`
/// bindings to `MmapOptions::new()` are seen (scoped to this block and its
/// children; not merged back into the caller's scope, matching how `guarded`
/// itself is not threaded back out of a nested block's own control-flow
/// forms).
fn walk_block(
    block: &Block,
    guarded_in: bool,
    ctx: &WalkCtx,
    sites: &mut Vec<SiteObservation>,
    opaque: &mut Vec<String>,
) -> bool {
    let mut guarded = guarded_in;
    let mut ctx = ctx.clone();
    for stmt in &block.stmts {
        guarded = match stmt {
            Stmt::Local(local) => {
                let mut guarded_after = guarded;
                if let Some(init) = &local.init {
                    guarded_after = walk_expr(&init.expr, guarded_after, &ctx, sites, opaque);
                    if let syn::Pat::Ident(pat_ident) = &local.pat
                        && let Expr::Call(call) = &*init.expr
                        && is_mmap_options_new_call(call)
                    {
                        ctx.bound_mmap_options.insert(pat_ident.ident.to_string());
                    }
                    if let Some((_, diverge)) = &init.diverge {
                        // `let ... else { diverge }`: the diverge block only
                        // runs when the pattern does NOT match, so -- like an
                        // `if`/`match` arm -- it is internal-only and never
                        // propagated to the statements that follow.
                        walk_expr(diverge, guarded_after, &ctx, sites, opaque);
                    }
                }
                guarded_after
            }
            Stmt::Expr(expr, _) => walk_expr(expr, guarded, &ctx, sites, opaque),
            // A nested item (e.g. `fn inner() { .. }`) is discovered and
            // walked separately, as its own entry, by
            // `collect_nested_function_items` -- it is a distinct callable
            // unit that does not run inline with these statements, so it is
            // correctly a no-op here rather than something to recurse into.
            Stmt::Item(_) => guarded,
            Stmt::Macro(stmt_macro) => {
                if !ctx.in_deferred_scope && !is_transparent_macro(&stmt_macro.mac.path) {
                    opaque.push(format!(
                        "{}: a macro invocation (`{}!(..)`) in statement position is opaque to \
                         this AST walk",
                        ctx.location,
                        path_to_string(&stmt_macro.mac.path)
                    ));
                }
                guarded
            }
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
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::or_short_circuit");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
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
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::and_short_circuit");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
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

/// F1 regression: `guard(..).is_ok() || unsafe { Mmap::map(&file) }` runs the
/// mmap exactly when the guard's `.is_ok()` is `false` -- i.e. exactly when
/// the guard FAILED. Visiting the raw guard call inside `.is_ok()` must not
/// be credited: only a form that propagates the guard's success
/// (`?`/`.expect`/`.unwrap`) may.
#[test]
fn walk_expr_does_not_credit_a_guard_call_collapsed_to_bool_via_is_ok_in_or() {
    let block: Block = syn::parse_str(
        r#"{
            reject_if_open_mmap_file_untrusted(&file, path).is_ok() || unsafe { Mmap::map(&file) };
        }"#,
    )
    .expect("parse F1 fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f1_is_ok_or");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly one Mmap::map() site"
    );
    assert!(
        !sites[0].dominated_by_guard,
        "`guard(..).is_ok() || map(..)` runs the map exactly when the guard FAILED; the guard \
         call being textually present must not be credited"
    );
}

/// F2 regression: `if guard(..).is_err() { unsafe { Mmap::map(&file) }; }`
/// runs the mmap exactly in the branch where the guard failed. Same root
/// cause as F1.
#[test]
fn walk_expr_does_not_credit_a_guard_call_collapsed_to_bool_via_is_err_in_if() {
    let block: Block = syn::parse_str(
        r#"{
            if reject_if_open_mmap_file_untrusted(&file, path).is_err() {
                unsafe { Mmap::map(&file) };
            }
        }"#,
    )
    .expect("parse F2 fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f2_is_err_if");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly one Mmap::map() site"
    );
    assert!(
        !sites[0].dominated_by_guard,
        "`if guard(..).is_err() {{ map(..) }}` runs the map exactly when the guard FAILED; the \
         guard call being textually present in the condition must not be credited"
    );
}

/// Control for F1/F2: the real idiom this crate's loaders use --
/// `crate::weights::mmap_trust::guard(..).map_err(..)?;` followed by the
/// construction -- must still be credited. Proves the fix distinguishes
/// "propagated via `?`" from "collapsed to bool", rather than simply never
/// crediting a guard call again.
#[test]
fn walk_expr_still_credits_a_guard_call_propagated_via_try_and_map_err() {
    let block: Block = syn::parse_str(
        r#"{
            crate::weights::mmap_trust::reject_if_open_mmap_file_untrusted(&file, path)
                .map_err(Into::into)?;
            unsafe { Mmap::map(&file) }
        }"#,
    )
    .expect("parse guard-then-map control fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::guard_then_map_control");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly one Mmap::map() site"
    );
    assert!(
        sites[0].dominated_by_guard,
        "`guard(..).map_err(..)?;` propagates the guard's success via `?` and must still be \
         credited to the construction call that follows"
    );
}

/// Control for F1/F2, `.expect`/`.unwrap` form: `guard(..).expect(msg);`
/// panics on failure, so continuation past it guarantees success -- the
/// pattern this crate's own `mmap_trust.rs` tests use (bare, since they are
/// scanned with `in_guard_definition_module: true`).
#[test]
fn walk_expr_still_credits_a_guard_call_propagated_via_expect() {
    let block: Block = syn::parse_str(
        r#"{
            let (file, meta) = open_trusted_mmap_file(&path).expect("fixture");
            unsafe { MmapOptions::new().map(&file) }
        }"#,
    )
    .expect("parse expect() control fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = WalkCtx {
        in_guard_definition_module: true,
        ..empty_ctx("fixture::guard_expect_control")
    };
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly one MmapOptions::new().map() site"
    );
    assert!(
        sites[0].dominated_by_guard,
        "`guard(..).expect(..)` panics on failure, so it must still be credited to the \
         construction call that follows"
    );
}

/// F3 regression: `MmapOptions::new()` bound to a local before `.map(..)` is
/// called on it must still be discovered as a construction site -- and,
/// being unguarded here, flagged as a violation.
#[test]
fn walk_expr_discovers_mmap_options_map_call_through_a_local_binding() {
    let block: Block = syn::parse_str(
        r#"{
            let options = MmapOptions::new();
            unsafe { options.map(&file) };
        }"#,
    )
    .expect("parse F3 fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f3_bound_options");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "a `MmapOptions::new()` bound to a local before `.map(..)` must still be discovered as \
         a construction site, not silently vanish"
    );
    assert_eq!(sites[0].selector, "MmapOptions::new().map()");
    assert!(
        !sites[0].dominated_by_guard,
        "the fixture has no guard call at all; the site must be reported unguarded"
    );
}

/// F5 regression: a locally shadowed function sharing the guard's bare name,
/// defined and called unqualified inside a module that is NOT
/// `weights::mmap_trust`, must not satisfy the guard-call requirement --
/// only [`GUARD_DEFINITION_MODULE`]'s own bare calls may.
#[test]
fn walk_expr_does_not_credit_a_locally_shadowed_guard_function() {
    let block: Block = syn::parse_str(
        r#"{
            fn reject_if_open_mmap_file_untrusted(
                _file: &std::fs::File,
                _path: &std::path::Path,
            ) -> Result<(), String> {
                Ok(())
            }
            reject_if_open_mmap_file_untrusted(&file, path)?;
            unsafe { Mmap::map(&file) }
        }"#,
    )
    .expect("parse F5 shadow fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    // `in_guard_definition_module: false` -- this fixture models a call site
    // OUTSIDE `weights/mmap_trust.rs`, where a bare call cannot be trusted.
    let ctx = empty_ctx("fixture::f5_bare_shadow");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly one Mmap::map() site"
    );
    assert!(
        !sites[0].dominated_by_guard,
        "a bare call to a locally shadowed function sharing the guard's name, outside the \
         guard's own defining module, must not be credited"
    );
}

/// F5 regression, qualified-path form: a decoy function reached through a
/// path whose LAST segment matches a guard name but whose full path is not
/// `weights::mmap_trust::..` must not satisfy the guard-call requirement.
#[test]
fn walk_expr_does_not_credit_a_wrongly_qualified_same_named_function() {
    let block: Block = syn::parse_str(
        r#"{
            mod decoy {
                pub(crate) fn reject_if_open_mmap_file_untrusted(
                    _file: &std::fs::File,
                    _path: &std::path::Path,
                ) -> Result<(), String> {
                    Ok(())
                }
            }
            decoy::reject_if_open_mmap_file_untrusted(&file, path)?;
            unsafe { Mmap::map(&file) }
        }"#,
    )
    .expect("parse F5 qualified-decoy fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f5_qualified_decoy");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly one Mmap::map() site"
    );
    assert!(
        !sites[0].dominated_by_guard,
        "`decoy::reject_if_open_mmap_file_untrusted` shares only the guard's last path segment; \
         it must not be credited"
    );
}

/// Control for F5: the crate-qualified path this crate's real callers use
/// must still be credited.
#[test]
fn walk_expr_still_credits_the_crate_qualified_guard_path() {
    let block: Block = syn::parse_str(
        r#"{
            crate::weights::mmap_trust::reject_if_open_mmap_file_untrusted(&file, path)
                .map_err(Into::into)?;
            unsafe { Mmap::map(&file) }
        }"#,
    )
    .expect("parse crate-qualified control fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f5_qualified_control");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "fixture must discover exactly one Mmap::map() site"
    );
    assert!(
        sites[0].dominated_by_guard,
        "`crate::weights::mmap_trust::reject_if_open_mmap_file_untrusted(..)` is the real \
         crate-qualified guard path and must be credited"
    );
}

/// F4 regression: an unguarded construction reachable only through
/// `Expr::RawAddr` (`&raw const`/`&raw mut`) must still be discovered.
#[test]
fn walk_expr_recurses_into_raw_addr_operand() {
    let block: Block = syn::parse_str(
        r#"{
            let _p = &raw const unsafe { Mmap::map(&file) };
        }"#,
    )
    .expect("parse RawAddr fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_raw_addr");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "an Mmap::map() call reachable only through the operand of `&raw const` must still be \
         discovered"
    );
    assert!(!sites[0].dominated_by_guard);
}

/// F4 regression: an unguarded construction reachable only through
/// `Expr::TryBlock` (`try { .. }`) must still be discovered.
#[test]
fn walk_expr_recurses_into_try_block() {
    let block: Block = syn::parse_str(
        r#"{
            let _r = try {
                unsafe { Mmap::map(&file) }?
            };
        }"#,
    )
    .expect("parse TryBlock fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_try_block");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "an Mmap::map() call reachable only through a `try {{ .. }}` block must still be \
         discovered"
    );
    assert!(!sites[0].dominated_by_guard);
}

/// F4 regression: an unguarded construction reachable only through
/// `Expr::Yield` must still be discovered.
#[test]
fn walk_expr_recurses_into_yield_operand() {
    let block: Block = syn::parse_str(
        r#"{
            yield unsafe { Mmap::map(&file) };
        }"#,
    )
    .expect("parse Yield fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_yield");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "an Mmap::map() call reachable only through the operand of `yield` must still be \
         discovered"
    );
    assert!(!sites[0].dominated_by_guard);
}

/// F4 regression: an unguarded construction inside an `async { .. }` block
/// must still be discovered (independently scoped, per `walk_expr`'s doc
/// comment -- it runs later, not inline -- but must not vanish outright).
#[test]
fn walk_expr_recurses_into_async_block() {
    let block: Block = syn::parse_str(
        r#"{
            let _fut = async {
                unsafe { Mmap::map(&file) }
            };
        }"#,
    )
    .expect("parse Async fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_async");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "an Mmap::map() call reachable only through an `async {{ .. }}` block must still be \
         discovered"
    );
    assert!(!sites[0].dominated_by_guard);
}

/// F4 regression: an unguarded construction inside a `const { .. }` block
/// (expression position) must still be discovered.
#[test]
fn walk_expr_recurses_into_const_block() {
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_const");
    let expr: Expr = syn::parse_str("const { unsafe { Mmap::map(&file) } }")
        .expect("parse Const expression fixture");
    walk_expr(&expr, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "an Mmap::map() call reachable only through a `const {{ .. }}` block must still be \
         discovered"
    );
    assert!(!sites[0].dominated_by_guard);
}

/// F4 regression: a macro invocation in expression position, in a function
/// that also has a real construction site, must be recorded as opaque --
/// the combination [`every_mmap_construction_site_is_guarded_or_explicitly_exempted`]
/// treats as a violation, since the macro's expansion could be hiding
/// another site or a guard call this walk cannot see.
#[test]
fn walk_expr_records_an_opaque_macro_invocation_alongside_a_real_site() {
    let block: Block = syn::parse_str(
        r#"{
            some_hiding_macro!();
            unsafe { Mmap::map(&file) };
        }"#,
    )
    .expect("parse Macro fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_macro");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "the real construction site must still be discovered"
    );
    assert_eq!(
        opaque.len(),
        1,
        "the macro invocation must be recorded as opaque now that the function also has a \
         discovered construction site"
    );
    assert!(opaque[0].contains("some_hiding_macro"));
}

/// Companion to the fixture above: a macro invocation is recorded as opaque
/// even when the function has NO construction site of its own -- the
/// decision about whether that matters belongs to the caller (see
/// `every_mmap_construction_site_is_guarded_or_explicitly_exempted`'s doc
/// comment: a function with no site is not flagged even though `opaque`
/// here is non-empty).
#[test]
fn walk_block_records_an_opaque_macro_statement() {
    let block: Block = syn::parse_str(
        r#"{
            some_hiding_macro!();
        }"#,
    )
    .expect("parse Stmt::Macro fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_stmt_macro");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert!(sites.is_empty());
    assert_eq!(opaque.len(), 1);
    assert!(opaque[0].contains("some_hiding_macro"));
}

/// Control for the two fixtures above: a macro invocation inside a
/// closure/`async`/`const` (deferred scope) must NOT be recorded as opaque,
/// since ordinary, harmless idioms like `.map_err(|e| format!(".."))` --
/// used throughout this crate's real guarded loaders -- are exactly this
/// shape, and would otherwise make every one of them unable to pass.
#[test]
fn walk_expr_does_not_record_a_macro_inside_a_deferred_closure_as_opaque() {
    let block: Block = syn::parse_str(
        r#"{
            crate::weights::mmap_trust::reject_if_open_mmap_file_untrusted(&file, path)
                .map_err(|e| format!("failed: {e}"))?;
            unsafe { Mmap::map(&file) }
        }"#,
    )
    .expect("parse deferred-macro control fixture");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_deferred_macro_control");
    walk_block(&block, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(sites.len(), 1);
    assert!(sites[0].dominated_by_guard);
    assert!(
        opaque.is_empty(),
        "a macro invocation inside a closure passed to `.map_err(..)` must not be recorded as \
         opaque -- it is deferred, not part of this function's own synchronous path"
    );
}

/// F4 regression: a nested `fn` item, invisible to `collect_functions`
/// before this fix, must be discovered as its own inventory entry.
#[test]
fn collect_functions_discovers_a_nested_fn_item() {
    let file: syn::File = syn::parse_str(
        r#"
        fn outer(file: &std::fs::File) {
            fn inner(file: &std::fs::File) {
                unsafe { Mmap::map(file) };
            }
            inner(file);
        }
        "#,
    )
    .expect("parse nested-fn fixture");
    let mut functions = Vec::new();
    let mut path_stack = Vec::new();
    collect_functions(&file.items, &mut path_stack, &mut functions);
    let inner = functions
        .iter()
        .find(|(path, _)| path == "outer::inner")
        .expect("nested `fn inner` must be discovered as its own entry (`outer::inner`)");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_nested_fn::outer::inner");
    walk_block(inner.1, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "the nested fn's own Mmap::map() call must be discovered"
    );
    assert!(!sites[0].dominated_by_guard);
}

/// F4 regression: a trait default method body, invisible to
/// `collect_functions` before this fix (only `Item::Fn`/`Item::Impl` were
/// handled), must be discovered as its own inventory entry.
#[test]
fn collect_functions_discovers_a_trait_default_method() {
    let file: syn::File = syn::parse_str(
        r#"
        trait Loader {
            fn load(&self, file: &std::fs::File) {
                unsafe { Mmap::map(file) };
            }
        }
        "#,
    )
    .expect("parse trait-default-method fixture");
    let mut functions = Vec::new();
    let mut path_stack = Vec::new();
    collect_functions(&file.items, &mut path_stack, &mut functions);
    let load = functions
        .iter()
        .find(|(path, _)| path == "Loader::load")
        .expect("trait default method must be discovered as its own entry (`Loader::load`)");
    let mut sites = Vec::new();
    let mut opaque = Vec::new();
    let ctx = empty_ctx("fixture::f4_trait_default::Loader::load");
    walk_block(load.1, false, &ctx, &mut sites, &mut opaque);
    assert_eq!(
        sites.len(),
        1,
        "the trait default method's own Mmap::map() call must be discovered"
    );
    assert!(!sites[0].dominated_by_guard);
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

        let in_guard_definition_module = relative == GUARD_DEFINITION_MODULE;

        let mut functions = Vec::new();
        let mut path_stack = Vec::new();
        collect_functions(&syntax.items, &mut path_stack, &mut functions);

        for (function_path, body) in functions {
            let location = format!("{relative}::{function_path}");
            let ctx = WalkCtx {
                location: &location,
                in_guard_definition_module,
                in_deferred_scope: false,
                bound_mmap_options: BTreeSet::new(),
            };
            let mut sites = Vec::new();
            let mut opaque = Vec::new();
            walk_block(body, false, &ctx, &mut sites, &mut opaque);
            if sites.is_empty() {
                // Nothing here for an opaque form to hide from this
                // contract's perspective -- see the module doc comment.
                continue;
            }

            // An opaque form (macro/Verbatim) anywhere in a function that
            // DOES construct an mmap can't be certified: it might expand to
            // hide another construction, or a guard call this walk would
            // otherwise have credited.
            for opaque_form in &opaque {
                violations.push(opaque_form.clone());
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
         MmapConstructionExemption -- or a function with a construction site that also \
         contains an opaque form this walk cannot certify:\n{}",
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
