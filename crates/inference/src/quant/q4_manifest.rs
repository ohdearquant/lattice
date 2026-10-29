//! Bounded reader for `quantize_index.json`, the per-tensor manifest written next to a Q4
//! checkpoint's `.q4`/`.f16` files, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-quant` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-quant` directly.

#[doc(inline)]
pub use lattice_quant::q4_manifest::{
    MAX_QUANTIZE_INDEX_LEN, ManifestFlavor, Q4Manifest, Q4ManifestEntry, Q4ManifestError,
    load_manifest, parse_manifest, read_manifest_bytes_bounded,
};
