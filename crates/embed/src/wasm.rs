//! wasm-bindgen JS bindings for browser-hosted embedding generation.
//!
//! Exposes a minimal surface: construct a [`LatticeEmbedder`] from in-memory
//! model bytes (no filesystem, no network access: the caller fetches or
//! otherwise holds the bytes on the JS side), then call [`LatticeEmbedder::embed`]
//! to get an L2-normalized embedding vector for a string.
//!
//! This wraps `lattice_inference::BertModel::from_bytes`, which itself avoids
//! `std::fs`/`mmap` entirely. That is required, not incidental:
//! `wasm32-unknown-unknown` has no real filesystem, and `memmap2`'s wasm
//! fallback compiles but every `Mmap::map` call returns
//! `io::ErrorKind::Unsupported` at runtime (see `SafetensorsFile::from_bytes`'s
//! doc comment in `lattice-inference`). Everything in this module runs
//! synchronously in memory: there is no `spawn_blocking`, disk cache, or
//! download path here; those belong to `NativeEmbeddingService` and stay
//! native-only.

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
