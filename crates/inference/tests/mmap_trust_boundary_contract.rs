//! Module-boundary contract for the mmap trust boundary (#1368/#1369).
//!
//! Three rounds hardened an AST walker that parsed this crate's own `src/`
//! with `syn`, found every mmap construction site, and asserted each was
//! dominated by a call to a guard in `src/weights/mmap_trust.rs`. Each round
//! closed real fail-open classes the walker's dominance analysis missed, and
//! each round's successor found new ones: a macro invocation whose expansion
//! might hide a construction site or a guard call, an `Expr::TryBlock` that
//! returned its inner guarded state directly, a `fn` item nested inside a
//! conditional body, a `MmapOptions::new()` bound to a local before `.map`
//! was called on it, and (latent, no real call site used the shape) a
//! `let mapper = Mmap::map; mapper(&file)` function-item alias. The walker
//! was trying to prove a property about **expression shapes**, and the space
//! of expression shapes that can reach a function call is not enumerable by
//! anything short of the compiler's own control-flow graph.
//!
//! This file replaces that walker with a property about **module
//! boundaries** instead, which is a small, countable space: [`weights::mmap_trust`]
//! is now the *only* place in this crate permitted to construct a memory map
//! (`memmap2::Mmap::map(..)` or `memmap2::MmapOptions::new()...map(..)`) from
//! a checkpoint file. Every production construction site that used to call
//! `Mmap::map`/`MmapOptions` directly (`quant/quarot/io.rs`,
//! `weights/f32_weights.rs`, `weights/q4_weights.rs`,
//! `forward/metal_qwen35.rs`) now calls one of `mmap_trust`'s own
//! `map_and_verify_trusted` / `map_after_untrusted_open` functions, which
//! fold the trust-boundary guard call and the map into a single function --
//! see their doc comments for why that makes "guarded before mapped"
//! structural rather than a per-call-site convention an external walker has
//! to keep reverifying. Storing an already-constructed mapping (a
//! `memmap2::Mmap` struct field, as `forward/moe_expert_cache.rs` and
//! `forward/metal_qwen35.rs`'s `Q3WeightBuf`/`Q4WeightBuf` types both do)
//! stays legal everywhere: construction is what this boundary protects, not
//! possession.
//!
//! # Why a plain per-line text scan, not another AST walk
//!
//! Every one of the five evasions the round-4 review constructed against the
//! old walker -- a macro-hidden call, a try-block, a nested `fn`, an unusual
//! `MmapOptions` binding, a function-item alias -- still has to **name**
//! `Mmap::map` or `MmapOptions` as literal characters somewhere in the file
//! that invokes it. Rust has no syntax for constructing a `memmap2` mapping
//! without spelling one of those two identifiers out. A macro invocation
//! expands *after* this scan runs, a `try` block is still lexically present
//! in the source, a nested `fn` is still text in the same file, a
//! `MmapOptions` bound to a local still contains the word `MmapOptions`, and
//! a function-item alias (`let mapper = Mmap::map;`) still writes `Mmap::map`
//! at the point it captures the function pointer. None of the AST shapes
//! that made the walker's *dominance* analysis hard have any bearing on
//! whether the *token* is present -- which is exactly what a per-line scan
//! checks, with no control-flow reasoning at all.
//!
//! # What this does NOT get for free
//!
//! - **A macro *defined* in this crate (or an external crate) whose
//!   expansion synthesizes `Mmap::map`/`MmapOptions` without either literal
//!   token appearing at the invocation site** would evade this scan the same
//!   way it would evade any textual tool (`grep`, `ast-grep` without macro
//!   expansion, etc.). This crate defines no `macro_rules!` of its own under
//!   `src/` that expands to either token (this scan reads raw source text,
//!   the same text a declarative macro's definition itself is written in, so
//!   such a macro's *definition* would still be caught even though a use of
//!   it elsewhere would not be) and takes no production dependency that
//!   could inject one into a checkpoint-loading path. Recorded here as a
//!   residual, not a silently assumed-away gap.
//! - **A `use memmap2::{Mmap as X, MmapOptions as Y}` rename** at a call site
//!   outside the boundary would let a caller invoke `X::map`/`Y::new()`
//!   without either token matching. No file in this crate imports either
//!   name under an alias today (verified by `grep -n 'use memmap2' -- every
//!   hit binds the bare names), and doing so is a far more conspicuous,
//!   review-visible act than any of the five AST evasions this design
//!   closes -- those hid inside completely idiomatic Rust; a rename import
//!   announces itself. Not defended against here; recorded as a residual
//!   for the same reason as the macro case above.
//! - **Ordering of the guard relative to the map** is no longer something
//!   this *test* checks at all -- it is enforced by `mmap_trust.rs`'s own
//!   function bodies (`map_and_verify_trusted`, `map_after_untrusted_open`
//!   each call the guard, unconditionally, as the first statement, before
//!   the `unsafe { Mmap::map(..) }` that follows). That is a stronger
//!   guarantee than the walker's dominance analysis ever gave -- a single
//!   straight-line function has no branches for a guard-only-on-one-arm bug
//!   to hide in -- but it now lives as an invariant of that module's source,
//!   not as something an external instrument re-derives on every run. A
//!   regression there would have to be caught by `mmap_trust.rs`'s own unit
//!   tests or code review, not by this file.
//!
//! # What is intentionally NOT preserved from the walker
//!
//! The walker's per-function inventory (`EXPECTED_MMAP_CONSTRUCTION_SITES`,
//! keyed by `mod`/`impl`/`fn` path) forced a conscious update whenever a
//! construction site was added, removed, or newly (un)guarded anywhere in
//! the crate. This contract needs no equivalent allowlist: the rule outside
//! `mmap_trust.rs` is simply "zero", unconditionally, so there is nothing to
//! enumerate or keep in sync. A new construction site anywhere outside the
//! boundary fails this test immediately, with no exemption list to remember
//! to update -- a *stronger* property than the walker's inventory gave, not
//! a weaker one dropped for convenience. Similarly dropped: the walker's own
//! regression fixtures for its dominance analysis (`||`/`&&` short-circuit,
//! guard-collapsed-to-bool, bare-vs-qualified guard names, opaque-macro
//! detection) tested the walker's *implementation*, not the crate's actual
//! security property -- with the walker deleted, there is nothing left for
//! them to regress-test.

use std::path::{Path, PathBuf};

/// The two lexical tokens that name lattice's mmap **construction** API:
/// `memmap2::Mmap::map(..)` and `memmap2::MmapOptions::new()...`. Naming the
/// *type* (`memmap2::Mmap` as a struct field, a function parameter or return
/// type) never matches either token -- `Mmap::map` requires the literal
/// `::map` suffix, `MmapOptions` names a different type entirely -- so
/// storing an already-constructed mapping stays unrestricted everywhere in
/// this crate; only building one outside the boundary is what these tokens
/// catch. See [`naming_the_mmap_type_without_constructing_one_stays_legal`]
/// for a direct proof of that distinction.
const CONSTRUCTION_TOKENS: &[&str] = &["Mmap::map", "MmapOptions"];

/// The sole file permitted to name a [`CONSTRUCTION_TOKENS`] token anywhere
/// in this crate's `src/` tree (code, doc comment, or otherwise -- see the
/// module doc comment's "plain per-line text scan" section for why this
/// contract does not distinguish them).
const TRUST_BOUNDARY_FILE: &str = "src/weights/mmap_trust.rs";

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

/// Every `(1-indexed line number, token)` pair found by scanning `source`
/// line by line for [`CONSTRUCTION_TOKENS`]. A plain substring scan, not an
/// AST walk -- see the module doc comment for why that is the point, not a
/// simplification made at the expense of soundness.
fn construction_token_hits(source: &str) -> Vec<(usize, &'static str)> {
    let mut hits = Vec::new();
    for (zero_indexed_line, line) in source.lines().enumerate() {
        for token in CONSTRUCTION_TOKENS {
            if line.contains(token) {
                hits.push((zero_indexed_line + 1, *token));
            }
        }
    }
    hits
}

/// The contract: outside [`TRUST_BOUNDARY_FILE`], no file under this crate's
/// `src/` names a mmap construction API -- and, in the same invocation, the
/// scan finds at least one such name *inside* that file. The second half is
/// the must-MATCH arm: a scan that finds nothing exits the same way whether
/// the crate is clean or the scan itself is broken (wrong root, wrong
/// extension filter, a typo'd token), and only a positive control run
/// alongside the negative one tells those apart.
#[test]
fn only_the_mmap_trust_boundary_names_a_construction_api() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_root = manifest_dir.join("src");

    let mut boundary_hits = 0usize;
    let mut violations: Vec<String> = Vec::new();

    for path in rust_sources_under(&src_root) {
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|reason| panic!("could not read {relative}: {reason}"));
        let hits = construction_token_hits(&source);

        if relative == TRUST_BOUNDARY_FILE {
            boundary_hits += hits.len();
            continue;
        }

        for (line_no, token) in hits {
            violations.push(format!(
                "{relative}:{line_no}: names `{token}`, a mmap construction API, outside \
                 {TRUST_BOUNDARY_FILE} -- every checkpoint mmap must be constructed inside that \
                 module (via map_and_verify_trusted / map_after_untrusted_open) so the \
                 trust-boundary guard cannot be skipped or misordered at the call site"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "mmap construction API named outside the trust boundary:\n{}",
        violations.join("\n")
    );

    assert!(
        boundary_hits > 0,
        "must-MATCH arm: the scan found zero `Mmap::map`/`MmapOptions` occurrences even inside \
         {TRUST_BOUNDARY_FILE} itself, which does construct mappings -- this means the scan (root, \
         extension filter, or token list) is broken, not that the crate is clean. A scan that \
         matches nothing exits the same way as a scan that never ran"
    );
}

/// Control for [`CONSTRUCTION_TOKENS`]: proves the policy is "construction is
/// banned outside the boundary", not "the word `Mmap` is banned outside the
/// boundary" -- a struct field storing an already-built `memmap2::Mmap`
/// (`forward/moe_expert_cache.rs::ExpertByteTable::mmap`,
/// `forward/metal_qwen35.rs`'s `Q3WeightBuf`/`Q4WeightBuf`) must not be
/// flagged. Synthetic fixture, not a real file, so this test's own use of
/// the word `Mmap` in a code sample does not depend on -- or drift with --
/// any real source file's exact contents.
#[test]
fn naming_the_mmap_type_without_constructing_one_stays_legal() {
    let fixture = "\
struct ExpertByteTable {
    mmap: memmap2::Mmap,
    payload_offset: u64,
}

fn take_mmap(_m: Option<memmap2::Mmap>) {}
";
    assert!(
        construction_token_hits(fixture).is_empty(),
        "storing or passing an already-constructed `memmap2::Mmap` must not be flagged -- only \
         `Mmap::map`/`MmapOptions`, the construction APIs, are banned outside the boundary"
    );
}

/// Control for the scan mechanics themselves: a file that does name a
/// construction token must be caught by [`construction_token_hits`]
/// directly, independent of the filesystem walk / boundary-file
/// carve-out logic [`only_the_mmap_trust_boundary_names_a_construction_api`]
/// layers on top. Isolates "the token scan works at all" from "the
/// boundary-file exemption is wired correctly".
#[test]
fn construction_token_hits_finds_both_tokens_and_reports_correct_line_numbers() {
    let fixture = "\
let file = std::fs::File::open(path)?;
let mmap = unsafe { memmap2::Mmap::map(&file) }?;
let opts = memmap2::MmapOptions::new();
";
    let hits = construction_token_hits(fixture);
    assert_eq!(
        hits,
        vec![(2, "Mmap::map"), (3, "MmapOptions")],
        "must find exactly the two construction-token lines, at their correct 1-indexed line \
         numbers, and nothing on the unrelated File::open line"
    );
}
