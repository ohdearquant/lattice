# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All lattice crates share one
workspace version; the workspace release notes are in
[`docs/releases/`](https://github.com/ohdearquant/lattice/tree/main/docs/releases).

## [Unreleased]

### Changed

- **Breaking:** `DriftReport` is `#[non_exhaustive]`. Code outside the crate can no longer
  build it with a struct literal or destructure it without `..`; reading its fields is unchanged.
  It gained a field in 0.11.0, and this lets later fields arrive without another breaking release.

## [0.12.1] - 2026-10-14

Documentation and package metadata only; no code changes.

### Documentation

- crates.io keywords and categories (#1936).

## [0.12.0] - 2026-10-13

No changes in this crate; the version follows the workspace.

## 0.1.0 to 0.11.1

These releases are described in the workspace release notes, `docs/releases/v<version>.md`.
