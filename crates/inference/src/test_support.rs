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

    /// Walks `root` for `.rs` files, panicking on any read error.
    ///
    /// Errors are loud on purpose. The obvious shape here skips an
    /// unreadable directory with `continue`, and that makes the whole guard
    /// fail open: one unreadable directory renders its entire subtree
    /// invisible, the scan finds nothing, and "no silent sites remain" is
    /// then a statement about a tree that was never read.
    fn rust_sources_under(root: &Path) -> Vec<PathBuf> {
        let mut pending = vec![root.to_path_buf()];
        let mut sources = Vec::new();
        while let Some(dir) = pending.pop() {
            let entries = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("cannot read {dir:?} while scanning for sources: {e}"));
            for entry in entries {
                let entry =
                    entry.unwrap_or_else(|e| panic!("cannot read an entry under {dir:?}: {e}"));
                let path = entry.path();
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
        for dir in ["src", "tests"] {
            let root = manifest_dir.join(dir);
            assert!(
                root.is_dir(),
                "{root:?} is not a directory; this scan cannot report an \
                 absence over a tree it did not find"
            );
            for path in rust_sources_under(&root) {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
                for line in silent_return_after_env_var(&source) {
                    all_hits.push(format!("{}:{line}", path.display()));
                }
                scanned.push(path);
            }
        }

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
}
