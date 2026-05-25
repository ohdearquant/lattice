# Release Checklist — v{VERSION}

**Date**: YYYY-MM-DD
**Previous**: v{PREV_VERSION}

## Changelog

### Breaking Changes
- (none, or list)

### New Features
- feat(crate): description — PR #N

### Fixes
- fix(crate): description — PR #N

### Internal
- chore/docs/refactor items that affect maintainers

## Pre-release

```sh
# 1. Ensure main is clean
git checkout main && git pull
git status  # must be clean

# 2. Version already bumped? Verify:
grep '^version' Cargo.toml  # should show {VERSION}
grep 'version = "' crates/*/Cargo.toml  # internal deps match

# 3. Full CI
make ci  # fmt + clippy + doc lint + test + release build

# 4. Dry-run publish (catches missing fields, version conflicts)
make publish-dry
```

## Publish

```sh
# 5. Tag
git tag -a v{VERSION} -m "v{VERSION}"
git push origin v{VERSION}

# 6. Publish to crates.io (leaf → embed → tune, with indexing waits)
make publish

# 7. GitHub release
gh release create v{VERSION} --title "v{VERSION}" --notes-file docs/releases/v{VERSION}.md
```

## Post-release

- [ ] Verify on crates.io: all 5 crates show v{VERSION}
- [ ] Smoke test: `cargo add lattice-inference@{VERSION}` in a fresh project
- [ ] Update getting-started.md if API changed
- [ ] Close relevant milestone/issues

## Rollback

If a crate publish is broken, yank and patch:
```sh
cargo yank lattice-{crate} --version {VERSION}
# fix, bump to {VERSION+patch}, re-publish
```
