# Implementation notes

- `crates/tune/tests/layer23_golden_bridge.rs` and its fixture preserve the
  removed layer-23 trainer's deterministic one-step output and compare the
  shared-driver replacement in an enforced macOS artifact job.
- `crates/tune/src/bin/train_common/full_driver.rs` now owns full training;
  `train_grad_full` and the deprecated `train_grad_layer23` compatibility
  command only resolve their respective CLI contracts.
- `crates/tune/src/lora/train.rs` derives the inclusive micro-LoRA suffix from
  `num_hidden_layers`, retaining the 24-layer range while covering 40- and
  64-layer model tops.
- Release golden bridge, focused unit tests, clippy, default-feature check,
  formatting, workflow lint, and documentation lint pass.

Bench disposition: N/A; lattice-tune trainers are outside the default
Criterion benchmark binaries.
