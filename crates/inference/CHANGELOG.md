# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Behavior change (#831):** `lattice serve`'s `/v1/chat/completions` now actually applies a
  request's `reasoning_budget` field to generation, instead of silently accepting and ignoring
  it. A malformed `reasoning_budget` (e.g. a string or object) is now rejected with
  `invalid_request_body`, matching `lattice_serve`'s pre-existing behavior for the same field.
  Clients that were previously sending `reasoning_budget` to `lattice serve` expecting no effect
  will now see it applied.
- **Behavior change (#831):** `lattice_serve`'s `/v1/chat/completions` now accepts a `stop`
  field (string or array of up to 4 non-empty strings) and applies it, instead of rejecting the
  request outright with `unsupported_feature` ("stop is not supported by this server"). Aligns
  with `lattice serve`'s pre-existing `stop` support.
- **Behavior change (#831):** both `lattice serve` and `lattice_serve` now reject a request
  whose `prompt + max_tokens + reasoning_budget` would leave no room for the generation-turn
  delimiter token: the shared context-window check now requires
  `prompt + max_new_tokens + reasoning_budget + 1 <= max_context` (previously `lattice serve`'s
  own HTTP preflight accepted `prompt + max_new_tokens == max_context` exactly, one token
  looser than this).
