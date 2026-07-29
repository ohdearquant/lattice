//! Guards against the `lattice-embed` native `lattice-inference` dependency
//! table losing the `f16` feature (#1095).
//!
//! `crates/embed/Cargo.toml` declares `lattice-inference` once now (the
//! native/wasm split that used to carry two independently-editable feature
//! lists was collapsed by #1094), and a standalone `cargo build -p
//! lattice-embed` resolves that one feature set in isolation — the same
//! resolve `cargo install lattice-embed` and the wasm/npm build get.
//!
//! Running the guard on that isolated resolve, rather than inside a
//! `--workspace` run, is deliberate. `lattice-tune`'s `lattice-inference`
//! dependency is optional but carries `features = ["f16"]`, so any invocation
//! that activates tune's `inference-hook` or `train-backward` alongside
//! `lattice-embed` lets Cargo's feature unification grant `lattice-embed` an
//! `f16`-enabled build for free, whatever `crates/embed/Cargo.toml` itself
//! asks for. Neither tune feature is in its defaults today, so a plain
//! `cargo test --workspace` does not currently mask anything; nothing enforces
//! that, and the isolated resolve does not depend on it staying true.
//!
//! This test never resolves through `--workspace`. It builds a half-precision
//! (BF16) tensor in memory and round-trips it through
//! `lattice_inference::weights::SafetensorsFile`, the same load path real
//! model weights go through (`weights/f32_weights.rs`). Without the `f16`
//! feature that load returns `InvalidSafetensors` instead of a tensor —
//! a compile-time-invisible, runtime-only failure, so a build-only check
//! would stay green on the very regression this test exists to catch.
//!
//! Run standalone to reproduce the isolation the guard depends on:
//!   `cargo test -p lattice-embed --test f16_feature_load_guard`

use lattice_inference::weights::SafetensorsFile;

/// Builds a minimal safetensors byte buffer containing a single BF16 tensor
/// named `weight` with two elements. Real model checkpoints carry BF16/F16
/// tensors identically shaped at the container level; only the byte width
/// and dtype tag matter to the load path under test, so a synthetic 4-byte
/// payload exercises the exact branch a real checkpoint would hit.
fn minimal_bf16_safetensors() -> Vec<u8> {
    // bf16 bit patterns for 1.0 and 2.0 (sign:1 exp:8 mantissa:7, top 16 bits
    // of the equivalent f32), little-endian on the wire like every other
    // safetensors dtype.
    let tensor_bytes: [u8; 4] = [0x80, 0x3f, 0x00, 0x40]; // [1.0_bf16, 2.0_bf16]

    let header = format!(
        r#"{{"weight":{{"dtype":"BF16","shape":[2],"data_offsets":[0,{}]}}}}"#,
        tensor_bytes.len()
    );
    let header_bytes = header.into_bytes();

    let mut buf = Vec::with_capacity(8 + header_bytes.len() + tensor_bytes.len());
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&tensor_bytes);
    buf
}

#[test]
fn bf16_tensor_loads_through_lattice_embeds_native_dependency() {
    let bytes = minimal_bf16_safetensors();
    let file = SafetensorsFile::from_bytes(bytes).expect(
        "safetensors container with a BF16 tensor must parse regardless of the f16 feature",
    );

    let (values, shape) = file.get_f32_tensor("weight").unwrap_or_else(|e| {
        panic!(
            "lattice-embed's native lattice-inference dependency was built \
             without the f16 feature, so a BF16 tensor hard-fails at load \
             instead of decoding to f32 (this is the #1094/#1095 regression \
             this guard exists to catch): {e}"
        )
    });

    assert_eq!(shape, [2]);
    assert!(
        (values[0] - 1.0).abs() < 1e-2,
        "bf16 1.0 decoded as {}",
        values[0]
    );
    assert!(
        (values[1] - 2.0).abs() < 1e-2,
        "bf16 2.0 decoded as {}",
        values[1]
    );
}
