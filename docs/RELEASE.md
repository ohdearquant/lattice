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

# 5. Package size against the crates.io upload limit, all five crates
scripts/package-size-check.sh
```

`make publish-dry` only validates the leaf tier (`lattice-fann`, `lattice-transport`) — Cargo
cannot dry-run a crate whose internal path dependencies are not yet live on the registry, so
`lattice-inference`, `lattice-embed`, and `lattice-tune` are not covered by the dry run.

### Package size

That same gap hides the registry's upload limit until publish time, which is the worst moment
to find it: crates.io rejects an archive over 10 MiB (10,485,760 bytes), and a rejection in
the second tier leaves the first tier already published and immutable.

`scripts/package-size-check.sh` covers all five crates and needs no registry access. It
brackets each archive from `cargo package --list`, which builds nothing and names exactly the
files that would ship, gzipped at level 6 — cargo's default, and the level to use here,
because level 9 shaves enough off the estimate to flatter a crate that is close to the limit.
`make publish` runs it before the first tier. Run it yourself as soon as a release branch
exists, because the remedy is a manifest change that goes through review like any other.

The limit is not theoretical for this workspace. `lattice-inference` published at 7.52 MiB at
v0.7.1 and 7.70 MiB at v0.9.0. At v0.10.0, two test-only tokenizer fixture directories — 43 MB
raw between them — took the measured archive to 10,493,367 bytes across 420 files, about 7.6 KB
over the limit. Adding both to the crate manifest's `exclude` list brought it to 3,010,538
bytes across 406 files with every other fixture still present. A published crate cannot run its
integration tests, so their fixtures are dead weight in the archive; `exclude` is the
instrument, not a smaller fixture.

Two of its behaviours are refusals rather than results, and both exit non-zero. It refuses when
`cargo package --list` fails, most often on a dirty working tree, which `cargo publish` will
refuse as well. It also refuses when a listed file is not on disk, which is what a measurement
taken from the wrong directory looks like: `cargo package --list` prints crate-relative paths,
so summing sizes from the workspace root finds almost nothing and would otherwise report a
comfortably small crate. Neither refusal is an absence of a problem.

## Normal Publish

```sh
# 6. Tag
git tag -a v{VERSION} -m "v{VERSION}"
git push origin v{VERSION}

# 7. Publish to crates.io in dependency-DAG order, with indexing waits
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
# 8. Create the tagged GitHub release as a draft
gh release create v{VERSION} --draft --title "v{VERSION}" --notes-file docs/releases/v{VERSION}.md

# 9. Dispatch the asset workflow from main; it verifies, uploads, and publishes the draft
gh workflow run release-binaries.yml --repo ohdearquant/lattice --ref main -f tag=v{VERSION}
```

Do not publish the draft manually while the asset workflow is running. Its draft-state checks are
separate API reads, not a lock against another actor publishing concurrently.
Publication during upload can leave the remote asset set partly or fully replaced before the
workflow notices and stops. The release can be published after the final state read and before the
workflow's publish edit. After a state-change or asset-verification failure, inspect the release
state and every remote asset before retrying.

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

# 3. Run the normal release gates and publish the replacement under a new tag and version:
git tag -a v{NEW_VERSION} -m "v{NEW_VERSION}"
git push origin v{NEW_VERSION}
make publish
gh release create v{NEW_VERSION} --draft --title "v{NEW_VERSION}" --notes-file docs/releases/v{NEW_VERSION}.md
gh workflow run release-binaries.yml --repo ohdearquant/lattice --ref main -f tag=v{NEW_VERSION}

# 4. Only after the replacement is live on crates.io and its GitHub asset workflow succeeds,
#    yank the broken version from every published crate:
for c in lattice-fann lattice-transport lattice-inference lattice-embed lattice-tune; do
  cargo yank --version {BROKEN_VERSION} "$c"
done

# 5. Verify crates.io reflects the yank, using the registry-check script in the
#    "Publishing" section of CLAUDE.md. Do not substitute a bare curl: crates.io
#    refuses a request that does not identify its caller and answers HTTP 403 with
#    a JSON error object, and code looking for a version in that object finds none,
#    so a refused read renders as "this version was never published". The script
#    runs `serde` first as a control for exactly that failure, and reads
#    `crate.max_stable_version` plus the per-version `yanked` booleans in
#    `versions[]`. There is no `latest_unyanked` field on the crate object.
```

A published GitHub release is not repaired in place by this workflow. Corrections always use the
new version, new tag, and new draft sequence above.

This is the same sequence used for the v0.2.3 release, which yanked the broken v0.2.2 (shipped
with a RoPE bug): new `cargo add` users got the fix directly, and existing users pinned to v0.2.2
received a yank warning on their next `cargo update`.
