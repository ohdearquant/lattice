# Release Process

This document is the maintainer-facing release process for the lattice workspace. It promotes
the checklist in [`docs/_templates/RELEASE.md`](_templates/RELEASE.md) into a concrete,
reconciled procedure. Commands and ordering here are grounded in the "Publishing" section of
`CLAUDE.md` and `scripts/publish.sh` — those two are the source of truth for the actual publish
order; if this document and either of them ever disagree, re-derive the order from
`crates/*/Cargo.toml` path dependencies rather than trusting stale prose.

## Publish Order

Publish order follows the internal dependency DAG, leaf crates first:

```
lattice-fann, lattice-transport   (leaf crates, no internal deps)
        │
        ▼  (wait for crates.io indexing)
lattice-inference                 (depends on lattice-fann via the `mixture` feature)
        │
        ▼  (wait for crates.io indexing)
lattice-embed, lattice-tune       (depend on lattice-inference / lattice-transport / lattice-fann)
```

This is the order implemented by `scripts/publish.sh` and run via `make publish`. When a feature
adds a new internal dependency (for example, `mixture` making `lattice-inference` depend on
`lattice-fann`), the publish order can change — re-derive it from `crates/*/Cargo.toml` path
dependencies rather than assuming the order above still holds.

## Pre-release

```sh
# 1. Ensure main is clean
git checkout main && git pull
git status  # must be clean

# 2. Version already bumped? Verify:
grep '^version' Cargo.toml               # should show {VERSION}
grep 'version = "' crates/*/Cargo.toml   # internal path deps match the workspace version
grep -nE 'lattice-embed = ' README.md    # README Quick Start pins track the release major.minor — bump on a minor release

# 3. Full CI
make ci  # fmt + clippy + doc lint + test + release build

# 4. Dry-run publish (catches missing fields, version conflicts)
make publish-dry
```

`make publish-dry` only validates the leaf tier (`lattice-fann`, `lattice-transport`) — Cargo
cannot dry-run a crate whose internal path dependencies are not yet live on the registry, so
`lattice-inference`, `lattice-embed`, and `lattice-tune` are not covered by the dry run.

## Normal Publish

```sh
# 5. Tag
git tag -a v{VERSION} -m "v{VERSION}"
git push origin v{VERSION}

# 6. Publish to crates.io in dependency-DAG order, with indexing waits
make publish
```

`make publish` runs `scripts/publish.sh`, which expands to:

```sh
cargo publish -p lattice-fann
cargo publish -p lattice-transport

sleep 30   # wait for crates.io indexing

cargo publish -p lattice-inference

sleep 30   # wait for crates.io indexing

cargo publish -p lattice-embed
cargo publish -p lattice-tune
```

```sh
# 7. GitHub release
gh release create v{VERSION} --title "v{VERSION}" --notes-file docs/releases/v{VERSION}.md
```

## Post-release

- [ ] Verify on crates.io: all five crates (`lattice-fann`, `lattice-transport`,
      `lattice-inference`, `lattice-embed`, `lattice-tune`) show `v{VERSION}`.
- [ ] Smoke test: `cargo add lattice-inference@{VERSION}` in a fresh project.
- [ ] Update `docs/getting-started.md` only if the public API changed.
- [ ] Close the relevant milestone/issues.

## Bump-and-Yank Recovery

crates.io versions are **immutable** — a broken publish cannot be overwritten or deleted, only
yanked. When a published release has a correctness bug, do **not** yank first. Yanking before a
fix is live leaves every consumer (including ones pinned to the exact broken version) with no
working version to resolve to. The required order is: ship the fix, then yank the broken version.

```sh
# 1. Bump the workspace version and internal path-dep versions to the next patch
#    (crates/*/Cargo.toml `version = "..."` fields must match the new workspace version).

# 2. Update the release notes file (rename if needed); add a
#    "Note on v<broken>" section explaining the bug and the yank.

# 3. Run the normal release gates and publish the replacement:
git tag -a v{NEW_VERSION} -m "v{NEW_VERSION}"
git push origin v{NEW_VERSION}
make publish
gh release create v{NEW_VERSION} --title "v{NEW_VERSION}" --notes-file docs/releases/v{NEW_VERSION}.md

# 4. Only after the replacement is live on crates.io, yank the broken version
#    from every published crate:
for c in lattice-fann lattice-transport lattice-inference lattice-embed lattice-tune; do
  cargo yank --version {BROKEN_VERSION} "$c"
done

# 5. Verify crates.io reflects the yank:
curl -s https://crates.io/api/v1/crates/{CRATE}
# should show latest_unyanked={NEW_VERSION} and yanked versions including {BROKEN_VERSION}
```

This is the same sequence used for the v0.2.3 release, which yanked the broken v0.2.2 (shipped
with a RoPE bug): new `cargo add` users got the fix directly, and existing users pinned to v0.2.2
received a yank warning on their next `cargo update`.
