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

# 7. Create the tagged GitHub release as a draft
gh release create v{VERSION} --draft --title "v{VERSION}" --notes-file docs/releases/v{VERSION}.md

# 8. Dispatch the asset workflow from main; it verifies, uploads, and publishes the draft
gh workflow run release-binaries.yml --repo ohdearquant/lattice --ref main -f tag=v{VERSION}
```

Do not publish the draft manually while the asset workflow is running. Its draft-state checks are
separate API reads, not a lock against another actor publishing concurrently.
Publication during upload can leave the remote asset set partly or fully replaced before the
workflow notices and stops. The release can be published after the final state read and before the
workflow's publish edit. After a state-change or asset-verification failure, inspect the release
state and every remote asset before retrying.

## Post-release

- [ ] Verify on crates.io: all 5 crates show v{VERSION}
- [ ] Smoke test: `cargo add lattice-inference@{VERSION}` in a fresh project
- [ ] Update getting-started.md if API changed
- [ ] Close relevant milestone/issues

## Rollback

If a published crate or GitHub asset is broken, publish a replacement under a new version and tag
before yanking the broken crate version:

```sh
# Fix, bump the workspace and internal path dependencies to {NEW_VERSION}, then run the gates.
git tag -a v{NEW_VERSION} -m "v{NEW_VERSION}"
git push origin v{NEW_VERSION}
make publish
gh release create v{NEW_VERSION} --draft --title "v{NEW_VERSION}" --notes-file docs/releases/v{NEW_VERSION}.md
gh workflow run release-binaries.yml --repo ohdearquant/lattice --ref main -f tag=v{NEW_VERSION}

# Only after the replacement is live and the asset workflow succeeds:
for c in lattice-fann lattice-transport lattice-inference lattice-embed lattice-tune; do
  cargo yank --version {VERSION} "$c"
done
```

Do not repair or reuse the published tag and version; corrections always use a new version, tag,
and draft.
