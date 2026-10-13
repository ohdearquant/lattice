# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All lattice crates share one
workspace version; the workspace release notes are in
[`docs/releases/`](https://github.com/ohdearquant/lattice/tree/main/docs/releases).

## [Unreleased]

## [0.12.0] - 2026-10-13

### Added

- `online-router`: update the gate from an explored mixture and refit one round with a leave-one-out baseline (#1917).

### Changed

- Every gate update entry refuses inputs that would write a non-finite value, and the gate write is all-or-nothing (#1917).

## 0.1.0 to 0.11.1

These releases are described in the workspace release notes, `docs/releases/v<version>.md`.
