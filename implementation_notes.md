Implementation Notes

- File changed: crates/inference/src/forward/metal_qwen35.rs
- In `mmap_q4_weight`, reordered operations: parse header, `file.metadata().len()` with `map_err(|e| format!("failed to stat {}: {e}", path.display()))`, `validate_q4_header_payload_bounds` using file length, then `MmapOptions::map`.
- No signature changes; no changes to `q4_weights.rs`.

Tests:
- `cargo test -p lattice-inference mmap_q4_weight_ --features metal-gpu,f16`
- Result: `3 passed; 0 failed; 0 ignored; 0 measured; 1536 filtered out`

Git:
- Commits on branch: `7633d97f1` then `94889c113`.
- Push to `origin/harden/540-q4-mmap-bounds` succeeded for both commits (`c657ef0fb..7633d97f1` and `7633d97f1..94889c113`).
