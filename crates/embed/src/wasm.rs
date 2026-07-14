//! Browser `wasm-bindgen` bindings for in-memory BERT embeddings and vector utilities.
//!
//! On `wasm32-unknown-unknown`, filesystem-backed loading and `mmap` are unsupported at
//! runtime: callers must provide model bytes in memory, and native download/cache paths do not
//! apply. This module runs synchronously. See `docs/design.md` for the WebAssembly boundary.

use lattice_inference::{BertModel, BertPooling};
use wasm_bindgen::prelude::*;

/// Install a panic hook that forwards Rust panics to the browser console via
/// `console.error`, instead of the opaque default
/// `"unreachable executed"` wasm trap message.
///
/// Call this once, before constructing a [`LatticeEmbedder`], if you want
/// readable panic diagnostics. Safe to call more than once (`set_once` is
/// idempotent) and safe to skip entirely.
#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// A loaded BERT-family embedding model, ready to embed text.
///
/// Constructed from in-memory model bytes, see [`LatticeEmbedder::new`].
/// Defaults to mean pooling; call [`LatticeEmbedder::use_cls_pooling`] for
/// BGE-family models, which expect CLS-token pooling instead (see
/// `lattice_inference::BertPooling`'s doc comment for the per-model-family
/// table).
#[wasm_bindgen]
pub struct LatticeEmbedder {
    model: BertModel,
}

#[wasm_bindgen]
impl LatticeEmbedder {
    /// Load a model from in-memory bytes.
    ///
    /// - `model_bytes`: the raw bytes of `model.safetensors`.
    /// - `config_bytes`: the UTF-8 bytes of `config.json`.
    /// - `tokenizer_bytes`: the UTF-8 bytes of `tokenizer.json`.
    ///
    /// Returns a JS exception (the `Err` arm, thrown by `wasm-bindgen`) if the
    /// bytes are not valid UTF-8 where text is expected, fail to parse, or
    /// don't describe a supported BERT-family model.
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_bytes: &[u8],
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
    ) -> Result<LatticeEmbedder, JsValue> {
        let config_json = std::str::from_utf8(config_bytes)
            .map_err(|e| JsValue::from_str(&format!("config.json is not valid UTF-8: {e}")))?;
        let tokenizer_json = std::str::from_utf8(tokenizer_bytes)
            .map_err(|e| JsValue::from_str(&format!("tokenizer.json is not valid UTF-8: {e}")))?;

        let model = BertModel::from_bytes(model_bytes.to_vec(), config_json, tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(LatticeEmbedder { model })
    }

    /// Switch to CLS-token pooling (the hidden state at position 0), the
    /// pooling strategy BGE v1.5 models expect. The default is mean pooling
    /// (attention-mask-weighted average), which is correct for E5 and MiniLM
    /// families. Call this immediately after construction for BGE models,
    /// before any `embed` call.
    #[wasm_bindgen(js_name = useClsPooling)]
    pub fn use_cls_pooling(&mut self) {
        self.model.set_pooling(BertPooling::CLS);
    }

    /// Embed a single text, returning its L2-normalized embedding vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, JsValue> {
        self.model
            .encode(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// The embedding dimensionality of the loaded model.
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }
}

// ---------------------------------------------------------------------------
// Vector-op bindings: expose `simd::*` directly to JS/wasm callers.
//
// Unrelated to `LatticeEmbedder` above (which wraps the BERT-family forward
// pass). These four are the `simd::*` khive ANN consumer contract
// (`dot_product`, `squared_euclidean_distance`, `cosine_similarity`,
// `normalize`; see `lib.rs`'s crate-level doc comment) made callable from JS,
// so a wasm/JS consumer benefits from the same SIMD128 kernels a native
// consumer gets, instead of falling back to a hand-rolled JS loop. Also the
// entry points `scripts/bench_wasm_simd.mjs` calls for its A/B measurement.
// ---------------------------------------------------------------------------

/// Dot product of two equal-length vectors via `simd::dot_product`.
///
/// Dispatches to the wasm32 SIMD128 kernel when this crate is built with
/// `-C target-feature=+simd128`, otherwise falls back to the scalar
/// implementation. Returns `0.0` if `a` and `b` have different lengths.
#[wasm_bindgen(js_name = simdDotProduct)]
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot_product(a, b)
}

/// Squared Euclidean (L2) distance between two equal-length vectors via
/// `simd::squared_euclidean_distance`.
///
/// Skips the final square root (see the Rust docs on
/// [`crate::simd::squared_euclidean_distance`] for the ordering invariant
/// this preserves). Returns `f32::MAX` if `a` and `b` have different lengths.
#[wasm_bindgen(js_name = simdSquaredEuclideanDistance)]
pub fn simd_squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::squared_euclidean_distance(a, b)
}

/// Cosine similarity of two equal-length, non-empty vectors via
/// `simd::cosine_similarity`.
///
/// Returns a value in `[-1.0, 1.0]`, or `0.0` for empty or mismatched-length
/// inputs.
#[wasm_bindgen(js_name = simdCosineSimilarity)]
pub fn simd_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::cosine_similarity(a, b)
}

/// L2-normalize a vector in place via `simd::normalize`.
///
/// Leaves the vector unchanged if its norm is zero or NaN (matches the
/// scalar reference; see `simd::normalize`'s doc comment).
#[wasm_bindgen(js_name = simdNormalize)]
pub fn simd_normalize(v: &mut [f32]) {
    crate::simd::normalize(v)
}

/// Reports whether the four `simd*` bindings above are dispatching to the
/// wasm32 SIMD128 kernels in *this* build.
///
/// Returns the exact same value the kernels themselves read to choose a
/// codepath (`crate::simd::simd_config().simd128_enabled()`), not a fresh
/// `cfg!(target_feature = "simd128")` re-derived here -- a binding-local
/// re-check could drift from the real dispatch condition without anyone
/// noticing. Exists so a two-build parity harness (one plain
/// `wasm32-unknown-unknown` build, one built with
/// `-C target-feature=+simd128`) can assert the SIMD128 build is actually
/// exercising the SIMD128 kernels instead of a stale or misconfigured
/// artifact silently falling back to scalar; see
/// `crates/embed/tests/wasm/simd128_parity_wasm.mjs`.
#[wasm_bindgen(js_name = simdSimd128Dispatch)]
pub fn simd_simd128_dispatch() -> bool {
    crate::simd::simd_config().simd128_enabled()
}
