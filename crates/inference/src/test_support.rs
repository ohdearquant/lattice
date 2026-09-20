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

    /// 1-based line numbers of `env::var(` calls that are followed, with no
    /// other line between, by a bare `return;` -- the shape this module
    /// exists to remove. Mirrors the `grep -rn -A1 'env::var(' | grep
    /// 'return;'` population search from issue #1664 exactly, so a hit here
    /// is a hit that search would find too.
    fn silent_return_after_env_var(source: &str) -> Vec<usize> {
        let lines: Vec<&str> = source.lines().collect();
        let mut hits = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line.contains("env::var(")
                && let Some(next) = lines.get(i + 1)
                && next.trim() == "return;"
            {
                hits.push(i + 1);
            }
        }
        hits
    }

    fn rust_sources_under(root: &Path) -> Vec<PathBuf> {
        let mut pending = vec![root.to_path_buf()];
        let mut sources = Vec::new();
        while let Some(dir) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let Ok(entry) = entry else { continue };
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
        for dir in ["src", "tests"] {
            let root = manifest_dir.join(dir);
            if !root.is_dir() {
                continue;
            }
            for path in rust_sources_under(&root) {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
                for line in silent_return_after_env_var(&source) {
                    all_hits.push(format!("{}:{line}", path.display()));
                }
            }
        }
        assert!(
            all_hits.is_empty(),
            "found env::var(...) immediately followed by a bare `return;` (a \
             silent-pass checkpoint site) at: {all_hits:?}. Route it through \
             require_checkpoint_dir instead."
        );
    }
}
