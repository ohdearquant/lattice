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
mod tests;
