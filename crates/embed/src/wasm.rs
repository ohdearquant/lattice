//! Browser `wasm-bindgen` bindings for in-memory BERT embeddings and vector utilities.
//!
//! On `wasm32-unknown-unknown`, filesystem-backed loading and `mmap` are unsupported at
//! runtime: callers must provide model bytes in memory, and native download/cache paths do not
//! apply. This module runs synchronously. See `docs/design.md` for the WebAssembly boundary.

use lattice_inference::{BertModel, BertPooling};
use wasm_bindgen::prelude::*;

/// Installs console-backed panic diagnostics for browser embeddings.
/// This optional, idempotent hook should run before constructing a [`LatticeEmbedder`].
/// See `docs/design.md` (§WebAssembly API details) for browser failure behavior.
#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// An in-memory BERT-family embedding model for browser use.
/// It defaults to mean pooling; BGE v1.5 callers must select CLS pooling before embedding.
/// See `docs/design.md` (§WebAssembly API details) for model and pooling requirements.
#[wasm_bindgen]
pub struct LatticeEmbedder {
    model: BertModel,
}

#[wasm_bindgen]
impl LatticeEmbedder {
    /// Loads a supported BERT-family model from safetensors, config, and tokenizer bytes.
    /// Returns a JavaScript exception for invalid JSON text, parsing failures, or unsupported models.
    /// See `docs/design.md` (§WebAssembly API details) for the in-memory loading boundary.
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

    /// Selects CLS-token pooling for BGE v1.5 models.
    /// Call this before embedding; mean pooling remains the default for E5 and MiniLM.
    /// See `docs/design.md` (§WebAssembly API details) for pooling selection.
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

// Expose the stable SIMD consumer contract to JavaScript — see docs/design.md.

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

/// Reports whether the vector bindings use this artifact's SIMD128 dispatch path.
/// Reads the dispatcher decision rather than independently re-deriving the build feature.
/// See `docs/design.md` (§WebAssembly API details) for parity-harness semantics.
#[wasm_bindgen(js_name = simdSimd128Dispatch)]
pub fn simd_simd128_dispatch() -> bool {
    crate::simd::simd_config().simd128_enabled()
}
