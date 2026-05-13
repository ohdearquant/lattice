# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2024-12-16

Initial release.

- Initial implementation of embedding generation
- SIMD-accelerated vector operations
- Local embedding support via fastembed (BGE-small default)
- LRU caching for embedding results with blake3 hashing
- Async embedding generation with Tokio runtime
- Benchmarks for SIMD operations and embedding performance
