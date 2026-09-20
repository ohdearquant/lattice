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

    #[test]
    fn no_silent_return_checkpoint_sites_remain_in_crate_source() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut all_hits = Vec::new();
        let mut scanned = Vec::new();
        let mut symlinks = Vec::new();
        // `benches` and `examples` are here because a checkpoint-reading site
        // does not care which Cargo target it compiles into, and these two
        // were outside the scan while carrying 56 `.rs` files between them.
        // None of the 56 carries the shape today, so this is prophylactic
        // rather than a fix -- which is the honest reason to land it, since
        // the population it protects is the one nobody is watching.
        let roots = ["src", "tests", "benches", "examples"];
        for dir in roots {
            let root = manifest_dir.join(dir);
            assert!(
                is_real_dir(&root),
                "{root:?} is not a real directory; this scan cannot report an \
                 absence over a tree it did not find, and a symlinked root \
                 would be followed out of the crate"
            );
            let walk = rust_sources_under(&root);
            symlinks.extend(walk.skipped_symlinks);
            let before = scanned.len();
            for path in walk.sources {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
                for line in silent_return_after_env_var(&source) {
                    all_hits.push(format!("{}:{line}", path.display()));
                }
                scanned.push(path);
            }
            // Per root, not just in total. A single large root keeps the
            // crate-wide count plausible while another contributes nothing,
            // and a root that silently reads empty is exactly the widening
            // failure this change is supposed to make visible.
            assert!(
                scanned.len() > before,
                "{root:?} contributed no .rs files; an absence over an empty \
                 root is not a clearance"
            );
        }

        assert!(
            symlinks.is_empty(),
            "the scan declined to follow {} symlink(s) under the scanned \
             roots, so the trees behind them were not read and the absence \
             below does not cover them: {symlinks:?}",
            symlinks.len()
        );

        // The population assert, without which an empty result is
        // indistinguishable from a walk that read nothing. A count alone is
        // weak -- a truncated walk still returns a plausible number -- so the
        // control names a file that must be in the set: this one.
        assert!(
            scanned.len() > 1,
            "the scan read {} file(s); an empty or near-empty walk makes the \
             absence below meaningless",
            scanned.len()
        );
        assert!(
            scanned.iter().any(|p| p.ends_with("test_support.rs")),
            "the scan did not reach this very file, so it did not read the \
             tree it claims to have cleared; scanned {} file(s)",
            scanned.len()
        );

        assert!(
            all_hits.is_empty(),
            "found env::var(...) immediately followed by a bare `return;` (a \
             silent-pass checkpoint site) at: {all_hits:?}. Route it through \
             require_checkpoint_dir instead."
        );
    }

    /// A dotted `(major, minor, patch)`.
    type Version = (u64, u64, u64);

    /// One `since` declaration: the 1-based line it was read from, and either
    /// its parsed version or the reason it could not be read.
    type SinceDeclaration = (usize, Result<Version, String>);

    /// Every `(major, minor, patch)` a `#[deprecated(since = ...)]` declares in
    /// this crate's sources, with the file and 1-based line it was read from.
    ///
    /// The scan is lexical and deliberately shaped so that it cannot see its
    /// own fixtures. It looks for the bare token followed by `=` and then an
    /// UNESCAPED double quote; every fixture in this module spells that quote
    /// `\"` inside a Rust string literal, so the bytes on disk are a backslash
    /// where the scanner requires a quote. That is the same property the
    /// silent-return scan above relies on, and it is stated here because it is
    /// load-bearing rather than incidental: without it this very file would
    /// report its own must-fail fixtures as crate defects and the guard could
    /// never be green.
    ///
    /// A declaration whose value does not parse as dotted integers is returned
    /// as `Err` rather than dropped. Skipping it would make an unreadable
    /// declaration indistinguishable from a compliant one, which is the
    /// fail-open direction: the caller turns it into a failure.
    fn deprecated_since_declarations(source: &str) -> Vec<SinceDeclaration> {
        let mut found = Vec::new();
        for (i, line) in source.lines().enumerate() {
            let mut rest = line;
            while let Some(at) = rest.find("since") {
                let after = &rest[at + "since".len()..];
                rest = after;
                // The token must stand alone: `licensed_since = "x"` is not a
                // deprecation attribute, and neither is `sincerely`.
                let before_ok = line[..line.len() - after.len() - "since".len()]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !c.is_alphanumeric() && c != '_');
                if !before_ok {
                    continue;
                }
                let after = after.trim_start();
                let Some(after) = after.strip_prefix('=') else {
                    continue;
                };
                let after = after.trim_start();
                let Some(after) = after.strip_prefix('"') else {
                    continue;
                };
                let Some(end) = after.find('"') else {
                    continue;
                };
                found.push((i + 1, parse_dotted_version(&after[..end])));
            }
        }
        found
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

    /// The scanner must SEE each spelling that appears in real attributes, and
    /// must NOT see the near-misses that share the word.
    ///
    /// Built as escaped one-line strings on purpose: see the note on
    /// [`deprecated_since_declarations`]. If a future edit spells one of these
    /// fixtures as a raw string or a multi-line literal, the quote stops being
    /// escaped on disk, this file starts reporting itself, and the failure will
    /// name the fixture rather than a real defect.
    #[test]
    fn deprecated_since_scanner_sees_each_spelling_and_no_near_miss() {
        let seen = |src: &str| -> Vec<Result<Version, String>> {
            deprecated_since_declarations(src)
                .into_iter()
                .map(|(_, v)| v)
                .collect()
        };

        assert_eq!(
            seen("#[deprecated(since = \"0.11.0\", note = \"x\")]"),
            vec![Ok((0, 11, 0))],
            "the rustfmt-normalised spelling must be seen"
        );
        assert_eq!(
            seen("    since=\"1.2.3\","),
            vec![Ok((1, 2, 3))],
            "the unspaced spelling must be seen; rustfmt normalises it today, \
             which is a formatting habit and not a guarantee"
        );
        assert_eq!(
            seen("    since   =   \"2.0\","),
            vec![Ok((2, 0, 0))],
            "a two-component version means patch 0"
        );
        assert_eq!(
            seen("let licensed_since = \"9.9.9\";"),
            Vec::new(),
            "a longer identifier ENDING in the token is not a deprecation \
             attribute; without the boundary check this scan invents defects"
        );
        assert_eq!(
            seen("/// deprecated since 0.11.0"),
            Vec::new(),
            "prose with no `= \"` is not a declaration"
        );
        assert!(
            matches!(seen("since = \"not-a-version\"").as_slice(), [Err(_)]),
            "an unparseable value is an ERROR, not a skip: dropping it would \
             make it indistinguishable from a compliant declaration"
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
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

        let mut offenders = Vec::new();
        let mut declarations = 0usize;
        let mut scanned = Vec::new();
        let mut symlinks = Vec::new();

        for dir in ["src", "tests", "benches", "examples"] {
            let root = manifest_dir.join(dir);
            assert!(
                is_real_dir(&root),
                "{root:?} is not a real directory; an absence over a tree that \
                 was not found is not a clearance, and a symlinked root would \
                 be followed out of the crate"
            );
            let walk = rust_sources_under(&root);
            symlinks.extend(walk.skipped_symlinks);
            let before = scanned.len();
            for path in walk.sources {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
                for (line, declared) in deprecated_since_declarations(&source) {
                    declarations += 1;
                    match declared {
                        Err(why) => offenders.push(format!("{}:{line}: {why}", path.display())),
                        Ok(v) if exceeds(current, v) => offenders.push(format!(
                            "{}:{line}: since = {v:?} is ahead of the crate version {current:?}",
                            path.display()
                        )),
                        Ok(_) => {}
                    }
                }
                scanned.push(path);
            }
            assert!(
                scanned.len() > before,
                "{root:?} contributed no .rs files; an absence over an empty \
                 root is not a clearance"
            );
        }

        assert!(
            symlinks.is_empty(),
            "the scan declined to follow {} symlink(s), so the trees behind \
             them were not read and the clearance below does not cover them: \
             {symlinks:?}",
            symlinks.len()
        );
        assert!(
            scanned.iter().any(|p| p.ends_with("test_support.rs")),
            "the scan did not reach this very file, so it did not read the \
             tree it claims to have cleared; scanned {} file(s)",
            scanned.len()
        );
        // The must-MATCH control, in the same pass that produces the absence.
        // A scanner whose pattern has rotted finds nothing and reports a clean
        // crate; this crate really does carry deprecation declarations, so a
        // zero here is an instrument failure and not a clearance.
        assert!(
            declarations > 0,
            "the scan read {} file(s) and found NO `since` declaration at all; \
             this crate carries several, so this is a dead scanner reporting a \
             clean result",
            scanned.len()
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
