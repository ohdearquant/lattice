# lattice-metal-runtime

Shared Metal buffer allocation, upload, command encoding and pipeline construction
for Lattice. The API is available on macOS with the opt-in `metal-gpu` feature.
The crate exposes no API on other targets or without that feature.

```toml
lattice-metal-runtime = { version = "0.14.0", features = ["metal-gpu"] }
```

Errors use `lattice-inference-core::InferenceError`. Scalar arguments require
`bytemuck::NoUninit` and a nonzero size smaller than 4096 bytes.

GPU tests that use these helpers live in `lattice-inference`, where they share
its measurement lock. This crate does not depend on the inference facade.
