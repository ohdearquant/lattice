//! Test-support helper for the checkpoint-gated unit tests in [`crate::model`].
//!
//! Thirteen `#[ignore]`d unit tests across [`crate::model::bert`],
//! [`crate::model::qwen`], and `crate::model::qwen35::generation` (a private
//! submodule, so not linked) need a real model checkpoint on disk and open by
//! reading an env var naming its directory. Reaching an `#[ignore]`d test needs an explicit `--ignored`,
//! which is a statement of intent to run it -- so an unset checkpoint there is
//! a provisioning failure, not a condition to tolerate silently. Before this
//! module existed, each site read the variable and `return`ed on failure with
//! no message at all: an explicitly requested run reported success for work
//! that never happened.
//!
//! [`require_checkpoint_dir`] is the panicking entry point the thirteen sites
//! call. It is deliberately a SECOND implementation of the checkpoint-refusal
//! idea, not a share of the resolver in
//! `crates/inference/tests/cpu_pre_migration_greedy_golden.rs`
//! (`resolve_model_dir`): that resolver has a different contract on purpose --
//! it reads two variables with a precedence rule and returns `Result` rather
//! than panicking, so its own controls can drive the refusal paths directly.
//! It also lives in an integration-test binary, which can only see this
//! crate's public surface, while this helper is `pub(crate)` and used from lib
//! unit tests. Forcing one shape onto both would either widen this crate's
//! public API for a test helper or weaken the sibling's contract. Two
//! implementations exist deliberately; if they are ever merged, the merge is
//! the change that decides which contract wins.

use std::path::{Path, PathBuf};

/// Resolves `var` to an existing absolute path, or explains why it could not.
///
/// Returned rather than panicked, mirroring the sibling resolver named in this
/// module's doc comment, so a control can drive every refusal path directly
/// without touching real process env state or the real filesystem.
fn resolve_checkpoint_dir(
    var: &str,
    lookup: impl Fn(&str) -> Option<String>,
    exists: impl Fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    let raw = lookup(var).ok_or_else(|| {
        format!(
            "{var} is unset. This test is #[ignore]d, so reaching it took an \
             explicit --ignored -- a statement of intent to run it. An \
             --ignored run without a checkpoint is a provisioning failure, \
             not a skip."
        )
    })?;
    let path = PathBuf::from(&raw);
    if !path.is_absolute() {
        return Err(format!(
            "{var}={raw:?} is relative; cargo test runs test binaries with the \
             crate directory as CWD, so a relative path would resolve \
             somewhere the caller did not mean. Pass an absolute path."
        ));
    }
    if !exists(&path) {
        return Err(format!(
            "{var}={raw:?} does not exist. An explicitly requested --ignored \
             run without a checkpoint is a provisioning failure, not a skip."
        ));
    }
    Ok(path)
}

/// Panics naming `var` if it is unset, relative, or does not exist; otherwise
/// returns the resolved absolute checkpoint directory. See the module doc
/// comment for why this is a second implementation rather than a share of the
/// integration-test sibling.
pub(crate) fn require_checkpoint_dir(var: &str) -> PathBuf {
    match resolve_checkpoint_dir(var, |v| std::env::var(v).ok(), Path::exists) {
        Ok(path) => path,
        Err(message) => panic!("{message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_refuses() {
        let err = resolve_checkpoint_dir("UNUSED", |_| None, |_| true).unwrap_err();
        assert!(err.contains("is unset"), "{err}");
    }

    #[test]
    fn relative_refuses() {
        let err =
            resolve_checkpoint_dir("VAR", |_| Some("relative/checkpoint".to_string()), |_| true)
                .unwrap_err();
        assert!(err.contains("is relative"), "{err}");
    }

    #[test]
    fn missing_refuses() {
        let err = resolve_checkpoint_dir(
            "VAR",
            |_| Some("/definitely/absolute/checkpoint".to_string()),
            |_| false,
        )
        .unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
    }

    /// The must-pass arm: without it, the three refusal controls above would
    /// all pass unconditionally against a helper that panicked (or, here,
    /// returned `Err`) regardless of input.
    #[test]
    fn present_absolute_resolves() {
        let dir = resolve_checkpoint_dir(
            "VAR",
            |_| Some("/definitely/absolute/checkpoint".to_string()),
            |_| true,
        )
        .unwrap();
        assert_eq!(dir, PathBuf::from("/definitely/absolute/checkpoint"));
    }

    // --- Guard against a fourteenth silent site (issue #1664) ---
    //
    // A unit test has full filesystem access at run time, so it CAN see a
    // lexical pattern across the crate's own checked-out source: it just has
    // to read the files. `CARGO_MANIFEST_DIR` is a compile-time absolute path
    // to this crate's root regardless of the process's CWD (the same
    // guarantee `crates/inference/tests/metal_measurement_lock_contract.rs`
    // already relies on for its own cross-file construction-site scan), so
    // this is not a best-effort heuristic scoped to wherever `cargo test`
    // happened to be invoked from.
    //
    // What it cannot do is stop a bad site from compiling in the first place;
    // it catches reintroduction the next time this test suite runs, which is
    // the same guarantee the rest of this crate's lexical-contract tests
    // (`stop_token_contract`, `metal_measurement_lock_contract`) already rely
    // on.

    /// 1-based line numbers of `env::var(` calls whose immediately following
    /// line returns without saying anything -- the shape this module exists
    /// to remove.
    ///
    /// The predicate is `trim().starts_with("return;")`, not equality. An
    /// earlier version compared for equality and its doc claimed to mirror
    /// the `grep -rn -A1 'env::var(' | grep 'return;'` population search
    /// "exactly". It did not: grep matches `return;` as a substring, so
    /// `        return; // provisioning` is a hit for the search that defined
    /// this population and was a miss for the scanner. A guard narrower than
    /// the search that motivated it is a guard that passes on the next
    /// instance, so the predicate is widened here rather than the claim
    /// softened, and the trailing-comment shape has its own fixture below.
    fn silent_return_after_env_var(source: &str) -> Vec<usize> {
        let lines: Vec<&str> = source.lines().collect();
        let mut hits = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line.contains("env::var(")
                && let Some(next) = lines.get(i + 1)
                && next.trim().starts_with("return;")
            {
                hits.push(i + 1);
            }
        }
        hits
    }

    /// What one walk of a root found: the `.rs` files, and every symlink it
    /// declined to follow.
    ///
    /// The second field exists because skipping a symlink and reporting a
    /// clean tree are the same observable. A symlinked source directory would
    /// be unscanned and the guard would still say "no silent sites remain",
    /// which is the fail-open direction. The caller asserts this is empty, so
    /// a symlink appearing under a scanned root stops the guard and hands the
    /// decision to a human instead of quietly shrinking the population.
    struct Walk {
        sources: Vec<PathBuf>,
        skipped_symlinks: Vec<PathBuf>,
    }

    /// Whether `path` is a directory in its own right, rather than a symlink
    /// that happens to point at one.
    ///
    /// This exists because [`rust_sources_under`] classifies every ENTRY from
    /// the directory entry's own file type and says so in its doc comment, and
    /// then its one caller validated each ROOT with `Path::is_dir`, which
    /// follows. The rule was stated in one place and broken in the other, so a
    /// symlinked `benches` would pass the root check, be walked, and contribute
    /// `.rs` files from outside the crate -- the same fail-open the entry-level
    /// classification exists to close, escaped one level up.
    fn is_real_dir(path: &Path) -> bool {
        std::fs::symlink_metadata(path).is_ok_and(|meta| meta.is_dir())
    }

    /// Walks `root` for `.rs` files, panicking on any read error.
    ///
    /// Errors are loud on purpose. The obvious shape here skips an
    /// unreadable directory with `continue`, and that makes the whole guard
    /// fail open: one unreadable directory renders its entire subtree
    /// invisible, the scan finds nothing, and "no silent sites remain" is
    /// then a statement about a tree that was never read.
    ///
    /// Symlinks are classified from the directory entry's own file type,
    /// never from `Path::is_dir`, which follows the link and answers about
    /// the target. Following is not a stylistic choice here: measured, a link
    /// at `src/x` pointing at `examples` made this scan report a hit at
    /// `src/x/probe.rs` for a file in `examples`, a root the scan had not
    /// named. A link pointing at an ancestor is worse, because the queue
    /// carries no visited set and the walk does not terminate.
    fn rust_sources_under(root: &Path) -> Walk {
        let mut pending = vec![root.to_path_buf()];
        let mut sources = Vec::new();
        let mut skipped_symlinks = Vec::new();
        while let Some(dir) = pending.pop() {
            let entries = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("cannot read {dir:?} while scanning for sources: {e}"));
            for entry in entries {
                let entry =
                    entry.unwrap_or_else(|e| panic!("cannot read an entry under {dir:?}: {e}"));
                let file_type = entry.file_type().unwrap_or_else(|e| {
                    panic!(
                        "cannot stat {:?} while scanning for sources: {e}",
                        entry.path()
                    )
                });
                let path = entry.path();
                if file_type.is_symlink() {
                    skipped_symlinks.push(path);
                } else if file_type.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    sources.push(path);
                }
            }
        }
        sources.sort();
        skipped_symlinks.sort();
        Walk {
            sources,
            skipped_symlinks,
        }
    }

    /// Deliberately-broken control: proves the scanner can FAIL by feeding it
    /// a fixture string carrying the exact bad shape, rather than only
    /// trusting that it passes over real source that is expected to be clean.
    #[test]
    fn silent_return_after_env_var_detects_the_known_bad_shape() {
        let fixture =
            "fn x() {\n    let Ok(v) = std::env::var(\"X\") else {\n        return;\n    };\n}\n";
        // Line 2 (1-based) is the `env::var(` call; line 3 is the bare
        // `return;` it is immediately followed by.
        assert_eq!(silent_return_after_env_var(fixture), vec![2]);
    }

    /// The shape that escaped the earlier equality predicate. A bare return
    /// carrying a trailing comment is still a silent pass -- the comment is
    /// in the source, not in the test output -- and the population search
    /// that defined this issue matches it.
    #[test]
    fn silent_return_after_env_var_detects_a_trailing_comment() {
        // On ONE physical source line, deliberately. A fixture describing this
        // shape must not LAY OUT as this shape: split across two lines with a
        // trailing `\`, the literal itself becomes an `env::var(` line followed
        // by a `return;` line, and the crate-wide scan below finds its own
        // fixture. It did, the first time this arm was written.
        let fixture = "fn x() {\n    let Ok(v) = std::env::var(\"X\") else {\n        return; // no checkpoint\n    };\n}\n";
        assert_eq!(silent_return_after_env_var(fixture), vec![2]);
    }

    /// The printed-skip shape (42 sites, explicitly out of scope for #1664)
    /// is a different, milder predicate and must not trip this scanner.
    #[test]
    fn silent_return_after_env_var_ignores_a_printed_skip() {
        let fixture = "fn x() {\n    let Ok(v) = std::env::var(\"X\") else {\n        \
                        eprintln!(\"skip\");\n        return;\n    };\n}\n";
        assert!(silent_return_after_env_var(fixture).is_empty());
    }

    /// One crate source file: its path, and its text read once.
    struct CrateSource {
        path: PathBuf,
        text: String,
    }

    /// Every `.rs` file under this crate's own roots, read once for the whole
    /// test binary, with the structural conditions that make an ABSENCE over
    /// them meaningful already asserted.
    ///
    /// Two scans depend on this population (the silent-return checkpoint scan
    /// and the deprecation-`since` contract), and before this helper existed
    /// each carried its own copy of the root list, the real-directory check,
    /// the per-root population assert and the skipped-symlink assert. That is
    /// the sibling-invocation-path shape this crate's own guidance warns
    /// about: a traversal fix would have to land twice or the two would drift.
    /// It is also two full reads of the same ~12 MB.
    ///
    /// The asserts live HERE rather than in the callers on purpose. A caller
    /// that forgot one would report an absence over a tree it had not
    /// established it read, and nothing would say so.
    fn validated_crate_sources() -> &'static [CrateSource] {
        static SOURCES: std::sync::OnceLock<Vec<CrateSource>> = std::sync::OnceLock::new();
        SOURCES.get_or_init(|| {
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let mut sources: Vec<CrateSource> = Vec::new();
            let mut symlinks = Vec::new();
            // `benches` and `examples` are here because a checkpoint-reading
            // site does not care which Cargo target it compiles into, and
            // these two were outside the scan while carrying 56 `.rs` files
            // between them.
            for dir in ["src", "tests", "benches", "examples"] {
                let root = manifest_dir.join(dir);
                assert!(
                    is_real_dir(&root),
                    "{root:?} is not a real directory; a scan cannot report an \
                     absence over a tree it did not find, and a symlinked root \
                     would be followed out of the crate"
                );
                let walk = rust_sources_under(&root);
                symlinks.extend(walk.skipped_symlinks);
                let before = sources.len();
                for path in walk.sources {
                    let text = std::fs::read_to_string(&path)
                        .unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
                    sources.push(CrateSource { path, text });
                }
                // Per root, not just in total. A single large root keeps the
                // crate-wide count plausible while another contributes
                // nothing, and a root that silently reads empty is exactly the
                // failure a widened scan is supposed to make visible.
                assert!(
                    sources.len() > before,
                    "{root:?} contributed no .rs files; an absence over an \
                     empty root is not a clearance"
                );
            }
            assert!(
                symlinks.is_empty(),
                "the walk declined to follow {} symlink(s) under the scanned \
                 roots, so the trees behind them were not read and no absence \
                 over this population covers them: {symlinks:?}",
                symlinks.len()
            );
            // A count alone is weak -- a truncated walk still returns a
            // plausible number -- so the control names a file that must be in
            // the set: this one.
            assert!(
                sources.iter().any(|s| s.path.ends_with("test_support.rs")),
                "the walk did not reach this very file, so it did not read the \
                 tree it claims to cover; read {} file(s)",
                sources.len()
            );
            sources
        })
    }

    #[test]
    fn no_silent_return_checkpoint_sites_remain_in_crate_source() {
        let mut all_hits = Vec::new();
        for source in validated_crate_sources() {
            for line in silent_return_after_env_var(&source.text) {
                all_hits.push(format!("{}:{line}", source.path.display()));
            }
        }

        assert!(
            all_hits.is_empty(),
            "found env::var(...) immediately followed by a bare `return;` (a \
             silent-pass checkpoint site) at: {all_hits:?}. Route it through \
             require_checkpoint_dir instead."
        );
    }

    /// A dotted `(major, minor, patch)`.
    type Version = (u64, u64, u64);

    /// One `#[deprecated(since = ...)]` declaration: the 1-based line it was
    /// read from, and either its parsed version or the reason it could not be
    /// used.
    type SinceDeclaration = (usize, Result<Version, String>);

    /// Every `since` a `#[deprecated(...)]` attribute declares in `source`.
    ///
    /// This parses the file with `syn` rather than matching text, and the
    /// first version of it did match text. That version was wrong in a way
    /// worth recording, because the lexical form looked sufficient: it
    /// required a plain `"` after `since =`, so `since = r"0.12.0"`,
    /// `since = r#"0.12.0"#` and a declaration split across lines all read as
    /// ABSENT. All three compile -- verified with `rustc --edition 2024` on a
    /// fixture carrying each -- so the cheapest way to silence that guard was
    /// to write a raw string, which is the fail-open direction for a guard
    /// whose entire job is to notice a new declaration.
    ///
    /// Parsing removes the class rather than the three instances. It also
    /// retires a property the lexical version depended on: it could not be
    /// allowed to see its own fixtures, so every fixture had to spell its
    /// quote `\"` inside a string literal. A string literal in a function body
    /// is not an attribute, so the fixtures below can be written as ordinary
    /// Rust.
    ///
    /// Two things are reported rather than skipped, because skipping either
    /// makes it indistinguishable from a compliant declaration: a value that
    /// is not a dotted integer version, and a `cfg_attr` that carries a
    /// `deprecated` payload this reader does not expand.
    fn deprecated_since_declarations(source: &str) -> Vec<SinceDeclaration> {
        use syn::spanned::Spanned;
        use syn::visit::Visit;

        struct Collect {
            found: Vec<SinceDeclaration>,
        }

        impl<'ast> Visit<'ast> for Collect {
            fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
                let line = attr.span().start().line;
                if attr.path().is_ident("cfg_attr")
                    && matches!(&attr.meta, syn::Meta::List(list)
                        if list.tokens.to_string().contains("deprecated"))
                {
                    self.found.push((
                        line,
                        Err("a cfg_attr carrying a `deprecated` payload is not \
                             expanded by this reader"
                            .to_string()),
                    ));
                    return;
                }
                if !attr.path().is_ident("deprecated") {
                    return;
                }
                // `#[deprecated]` and `#[deprecated = "why"]` declare no
                // `since` and are not this guard's subject; only the list form
                // can carry one.
                if !matches!(attr.meta, syn::Meta::List(_)) {
                    return;
                }
                let mut hit: Option<SinceDeclaration> = None;
                let parsed = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("since") {
                        let lit: syn::LitStr = meta.value()?.parse()?;
                        hit = Some((lit.span().start().line, parse_dotted_version(&lit.value())));
                    } else {
                        // Consume this key's value, whatever it is, so an
                        // unrelated key does not abort the walk over `since`.
                        let _ = meta
                            .value()
                            .and_then(syn::parse::ParseBuffer::parse::<syn::Expr>);
                    }
                    Ok(())
                });
                match (parsed, hit) {
                    (Ok(()), Some(found)) => self.found.push(found),
                    (Ok(()), None) => {}
                    (Err(e), _) => self.found.push((
                        line,
                        Err(format!("a `deprecated` attribute did not parse: {e}")),
                    )),
                }
            }
        }

        let file = match syn::parse_file(source) {
            Ok(file) => file,
            // Loud, not skipped. A file this reader cannot parse is a file
            // whose declarations it cannot see, and reporting it as clean
            // would be the same absence a compliant file produces.
            Err(e) => {
                return vec![(
                    e.span().start().line,
                    Err(format!("source did not parse: {e}")),
                )];
            }
        };
        let mut collect = Collect { found: Vec::new() };
        collect.visit_file(&file);
        collect.found
    }

    /// `"0.11"` and `"0.11.0"` both mean the same release; anything else is an
    /// error carrying the offending text, never a silent zero.
    fn parse_dotted_version(raw: &str) -> Result<Version, String> {
        let mut parts = raw.split('.');
        let mut next = || -> Result<u64, String> {
            match parts.next() {
                None => Ok(0),
                Some(p) => p
                    .parse::<u64>()
                    .map_err(|_| format!("{raw:?} is not a dotted integer version")),
            }
        };
        let (major, minor, patch) = (next()?, next()?, next()?);
        if parts.next().is_some() {
            return Err(format!("{raw:?} has more than three components"));
        }
        Ok((major, minor, patch))
    }

    /// The reader must see every spelling `rustc` accepts, and no near-miss.
    ///
    /// The three must-MATCH arms after the plain one are the ones that matter:
    /// each was a silent MISS for the lexical predecessor, and each compiles.
    /// They are the reason this reader parses instead of matching text.
    #[test]
    fn deprecated_since_reader_sees_every_accepted_spelling_and_no_near_miss() {
        let seen = |src: &str| -> Vec<Result<Version, String>> {
            deprecated_since_declarations(src)
                .into_iter()
                .map(|(_, v)| v)
                .collect()
        };

        assert_eq!(
            seen("#[deprecated(since = \"0.11.0\", note = \"x\")]\npub fn a() {}"),
            vec![Ok((0, 11, 0))],
            "the ordinary spelling must be seen"
        );
        assert_eq!(
            seen("#[deprecated(since = r\"0.12.0\")]\npub fn a() {}"),
            vec![Ok((0, 12, 0))],
            "a raw string must be seen: the lexical predecessor missed this \
             and it compiles, so it was the cheapest way to silence the guard"
        );
        assert_eq!(
            seen("#[deprecated(since = r#\"0.13.0\"#)]\npub fn a() {}"),
            vec![Ok((0, 13, 0))],
            "a hashed raw string must be seen for the same reason"
        );
        assert_eq!(
            seen("#[deprecated(\n    since\n        = \"0.14.0\"\n)]\npub fn a() {}"),
            vec![Ok((0, 14, 0))],
            "a declaration split across lines must be seen: a line-oriented \
             predicate cannot see this one at all"
        );
        assert_eq!(
            seen("#[deprecated(since = \"0.11.0\")]\npub type A = u8;"),
            vec![Ok((0, 11, 0))],
            "the attribute is not tied to one item kind"
        );

        assert_eq!(
            seen("pub fn a() { let licensed_since = \"9.9.9\"; }"),
            Vec::new(),
            "a longer identifier ending in the token is not an attribute"
        );
        assert_eq!(
            seen("/// deprecated since 0.11.0\npub fn a() {}"),
            Vec::new(),
            "prose is not a declaration"
        );
        assert_eq!(
            seen("pub fn a() { let s = \"#[deprecated(since = \\\"9.9.9\\\")]\"; }"),
            Vec::new(),
            "a string containing an attribute is a string. The lexical \
             predecessor needed every fixture escaped so it could not read \
             itself; parsing removes that requirement, and this arm pins it"
        );
        assert_eq!(
            seen("#[deprecated]\npub fn a() {}\n#[deprecated = \"why\"]\npub fn b() {}"),
            Vec::new(),
            "the bare and name-value forms declare no `since`"
        );

        assert!(
            matches!(
                seen("#[deprecated(since = \"nope\")]\npub fn a() {}").as_slice(),
                [Err(_)]
            ),
            "an unparseable value is an ERROR, not a skip: dropping it would \
             make it indistinguishable from a compliant declaration"
        );
        assert!(
            matches!(
                seen("#[cfg_attr(feature = \"x\", deprecated(since = \"9.9.9\"))]\npub fn a() {}")
                    .as_slice(),
                [Err(_)]
            ),
            "a cfg_attr carrying a deprecation is reported unresolved rather \
             than silently unread"
        );
        assert!(
            matches!(seen("this is not rust").as_slice(), [Err(_)]),
            "a file that does not parse is reported, never treated as clean"
        );
    }

    /// The comparison must reject a future version at every component, and
    /// accept the current one.
    ///
    /// The equal case is the arm with teeth. `since` naming the version that
    /// will first CONTAIN the deprecation is the correct declaration, so a
    /// guard written with `>=` would condemn every correct site and be
    /// reverted rather than fixed.
    #[test]
    fn a_since_above_the_crate_version_is_the_only_rejected_case() {
        let current: Version = (0, 11, 0);
        assert!(
            exceeds(current, (0, 11, 1)),
            "a future patch must be caught"
        );
        assert!(
            exceeds(current, (0, 12, 0)),
            "a future minor must be caught"
        );
        assert!(exceeds(current, (1, 0, 0)), "a future major must be caught");
        assert!(
            !exceeds(current, current),
            "the version that first contains the deprecation is the CORRECT \
             declaration and must pass"
        );
        assert!(!exceeds(current, (0, 10, 9)), "an older version must pass");
    }

    fn exceeds(current: Version, declared: Version) -> bool {
        declared > current
    }

    /// No `#[deprecated(since = ...)]` in this crate may name a version the
    /// crate has not reached.
    ///
    /// This exists because five declarations drifted to `0.11.1` against an
    /// unreleased `0.11.0` and nothing noticed; they were corrected by reading
    /// the registry by hand. A hand check that has to be remembered is not a
    /// guard.
    ///
    /// SCOPE, stated because the invariant is wider than the check: this reads
    /// `lattice-inference`'s own roots and compares against
    /// `CARGO_PKG_VERSION`, which every crate here inherits from
    /// `[workspace.package]`. `crates/embed` carries a `since` declaration that
    /// this scan does not reach, and reaching it would mean walking out of the
    /// crate, which is exactly the fail-open [`rust_sources_under`] refuses.
    /// So the uncovered population is one site in one sibling crate, named
    /// rather than implied.
    #[test]
    fn no_deprecated_since_exceeds_the_crate_version() {
        let current = parse_dotted_version(env!("CARGO_PKG_VERSION"))
            .expect("the crate's own version must parse");

        let mut offenders = Vec::new();
        let mut declarations = 0usize;
        let sources = validated_crate_sources();
        for source in sources {
            for (line, declared) in deprecated_since_declarations(&source.text) {
                declarations += 1;
                match declared {
                    Err(why) => offenders.push(format!("{}:{line}: {why}", source.path.display())),
                    Ok(v) if exceeds(current, v) => offenders.push(format!(
                        "{}:{line}: since = {v:?} is ahead of the crate version {current:?}",
                        source.path.display()
                    )),
                    Ok(_) => {}
                }
            }
        }

        // The must-MATCH control, in the same pass that produces the absence.
        // A reader that has stopped seeing attributes finds nothing and
        // reports a clean crate; this crate really does carry deprecation
        // declarations, so a zero here is an instrument failure.
        assert!(
            declarations > 0,
            "the scan read {} file(s) and found NO `since` declaration at all; \
             this crate carries several, so this is a dead reader reporting a \
             clean result",
            sources.len()
        );

        assert!(
            offenders.is_empty(),
            "deprecation `since` declarations this crate cannot support or \
             cannot read, against crate version {current:?}: {offenders:?}. A \
             `since` names the first release CONTAINING the deprecation, so it \
             is at or below the current version, never ahead of it; a value \
             that does not parse is reported here rather than skipped."
        );
    }

    /// `is_real_dir` must reject a symlink that points at a directory, and the
    /// naive form must accept it.
    ///
    /// The second assertion is the one that gives this test teeth. Without it,
    /// swapping `is_real_dir` back for `Path::is_dir` leaves the test green on
    /// the real-directory case alone, and a guard test that cannot express the
    /// defect it guards is decoration. Asserting that `link.is_dir()` is TRUE
    /// pins the difference between the two forms rather than the behaviour of
    /// the one we happen to call.
    #[test]
    fn is_real_dir_rejects_a_symlink_to_a_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real");
        std::fs::create_dir(&real).expect("create real dir");
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");

        assert!(is_real_dir(&real), "a real directory must be accepted");
        assert!(
            !is_real_dir(&link),
            "a symlink pointing at a directory must be rejected: following it \
             is what lets a scanned root reach outside the crate"
        );
        assert!(
            link.is_dir(),
            "control: Path::is_dir must ACCEPT this symlink, otherwise this \
             fixture cannot tell the two forms apart and proves nothing"
        );
        assert!(
            !is_real_dir(&tmp.path().join("absent")),
            "a path that does not exist is not a real directory"
        );
    }
}
