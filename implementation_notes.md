Implementation Notes

- In `crates/inference/src/forward/metal_qwen35.rs::mmap_q4_weight`, reordered operations to validate Q4 payload bounds before calling `memmap2::MmapOptions::new().map(&file)`: open file → parse header → `file.metadata().len()` with `failed to stat` error mapping → bounds validation against file length → mmap call.
- Did not modify function signatures, error text formats, or `q4_weights.rs`.

Tests run:
- `cargo test -p lattice-inference mmap_q4_weight_ --features metal-gpu,f16`
- Result: `3 passed; 0 failed; 0 ignored; 0 measured; 1536 filtered out` (covered tests: `mmap_q4_weight_rejects_truncated_block_payload_before_dispatch`, `mmap_q4_weight_rejects_payload_one_byte_short`, `mmap_q4_weight_accepts_fully_populated_payload`).

Push confirmation:
- To be added after successful commit and push.
