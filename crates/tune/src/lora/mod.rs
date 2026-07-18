//! LoRA adapters, lightweight training helpers, and inference integration.
//!
//! A layer stores row-major `A: (rank, d_in)` and `B: (d_out, rank)` and adds
//! `(alpha / rank) * B @ (A @ x)` to a base projection. The adapter constructor
//! rejects non-finite alpha values; a zero rank has an effective scale of zero.
//!
//! See `docs/lora-core.md` for the training, blending, and manifest design.

mod apply;
pub mod blend;
#[cfg(all(feature = "safetensors", feature = "serde"))]
pub mod loader;
#[cfg(feature = "serde")]
pub mod manifest;
pub mod online;
pub mod optimizer;
#[cfg(feature = "mixture")]
pub mod router_update;
#[cfg(feature = "safetensors")]
mod safetensors;
#[cfg(feature = "train-backward")]
pub mod train;
#[cfg(feature = "train-backward")]
#[doc(hidden)]
pub mod train_core;

pub use apply::apply_lora;
pub use blend::blend_lora_adapters;
#[cfg(all(feature = "safetensors", feature = "serde"))]
pub use loader::{LoadedAdapter, RunningRevisions};
#[cfg(feature = "serde")]
pub use manifest::{AdapterId, LoraManifest, ManifestEntry};
pub use online::{AdaptStepResult, adapt_step};
pub use optimizer::{AdamState, LoraGradients, compute_lora_gradients};
#[cfg(feature = "safetensors")]
pub use safetensors::{AdapterGovernance, load_peft_safetensors, save_peft_safetensors};

use std::collections::HashMap;
use std::path::Path;

/// LoRA configuration parameters.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Low-rank dimension. Typical values: 4, 8, 16, 32, 64.
    pub rank: usize,
    /// Scaling factor. The effective scale is `alpha / rank`.
    /// When `alpha == rank`, the scale is 1.0 (no extra scaling).
    pub alpha: f32,
    /// Names of the modules that have LoRA adapters.
    /// e.g., `["q_proj", "v_proj", "gate_proj", "up_proj"]`
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    /// Compute the LoRA scaling factor: `alpha / rank`.
    pub fn scale(&self) -> f32 {
        let scale = if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        };
        if self.alpha.is_finite() && scale.is_finite() {
            scale
        } else {
            0.0
        }
    }

    /// Validate that the LoRA alpha and effective scale are finite.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::TuneError::Validation`] when `alpha` or the
    /// effective `alpha / rank` scale is not finite.
    pub fn validate(&self) -> crate::error::Result<()> {
        if !self.alpha.is_finite() {
            return Err(crate::error::TuneError::Validation(format!(
                "LoRA alpha must be finite, got {}",
                self.alpha
            )));
        }
        let scale = if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        };
        if !scale.is_finite() {
            return Err(crate::error::TuneError::Validation(format!(
                "LoRA effective scale must be finite, got {scale}"
            )));
        }
        Ok(())
    }
}

/// A row-major low-rank update for one linear projection.
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// Matrix A, row-major `(rank, d_in)`.
    pub a: Vec<f32>,
    /// Matrix B, row-major `(d_out, rank)`.
    pub b: Vec<f32>,
    /// Input dimension of the base linear projection.
    pub d_in: usize,
    /// Output dimension of the base linear projection.
    pub d_out: usize,
    /// LoRA rank (inner dimension).
    pub rank: usize,
}

/// A validated collection of LoRA layers keyed by transformer layer and module.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#adapter-validation) for serving-time validation.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Adapter configuration.
    config: LoraConfig,
    /// Per-layer, per-module LoRA weights.
    /// Key: `(layer_idx, module_name)` e.g. `(5, "q_proj")`.
    layers: HashMap<(usize, String), LoraLayer>,
}

impl LoraAdapter {
    /// Load a LoRA adapter from a PEFT-format safetensors file.
    /// Returns an error for invalid tensors, pairs, or ranks.
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#adapter-validation) for the loading boundary.
    #[cfg(feature = "safetensors")]
    pub fn from_safetensors(path: &Path) -> crate::error::Result<Self> {
        safetensors::load_peft_safetensors(path)
    }

    /// Save this LoRA adapter to a PEFT-format safetensors file.
    ///
    /// `governance` optionally adds provenance metadata to the file header.
    /// Returns an error when serialization or writing fails.
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#adapter-validation) for the metadata boundary.
    #[cfg(feature = "safetensors")]
    pub fn save_safetensors(
        &self,
        path: &Path,
        governance: Option<&safetensors::AdapterGovernance>,
    ) -> crate::error::Result<()> {
        safetensors::save_peft_safetensors(self, path, governance)
    }

    /// Construct an adapter from pre-built components (for testing or
    /// when loading from a custom format).
    ///
    /// Every non-empty layer's `a`/`b` buffers must be sized exactly
    /// `rank * d_in` and `d_out * rank` (the layout `apply_lora` indexes
    /// into); a layer with BOTH `a` and `b` empty is an untrained/placeholder
    /// module and is exempt (mirrors `save_peft_safetensors`, which skips
    /// the same layers). A layer with exactly one of `a`/`b` empty is
    /// malformed, not a placeholder, and is rejected like any other
    /// length mismatch. This is the single construction chokepoint
    /// (safetensors loading, blending, and training all route through it),
    /// so downstream code — including
    /// [`validate_against`](Self::validate_against) and `apply` — can rely
    /// on the invariant without re-checking it. `apply_lora` itself also
    /// verifies its input against the layer's declared geometry before
    /// indexing, as a second boundary for adapter data built by other means.
    ///
    /// # Errors
    ///
    /// Returns an error when the adapter configuration is invalid, or when
    /// any non-placeholder layer's `a`/`b` buffer length doesn't match its
    /// own declared `rank`/`d_in`/`d_out`.
    pub fn new(
        config: LoraConfig,
        layers: HashMap<(usize, String), LoraLayer>,
    ) -> crate::error::Result<Self> {
        config.validate()?;
        for ((layer_idx, module), layer) in &layers {
            // Both `a` and `b` empty denotes an untrained/not-yet-populated
            // module (see `save_peft_safetensors`, which skips these the
            // same way) and is exempt from the buffer/rank check below.
            // Exactly one empty is not a valid placeholder state and falls
            // through to the length checks, which reject it.
            if layer.a.is_empty() && layer.b.is_empty() {
                continue;
            }
            let expected_a = layer.rank.checked_mul(layer.d_in).ok_or_else(|| {
                crate::error::TuneError::Validation(format!(
                    "LoRA layer {layer_idx} module '{module}': rank*d_in overflowed usize \
                     (rank={}, d_in={})",
                    layer.rank, layer.d_in
                ))
            })?;
            if layer.a.len() != expected_a {
                return Err(crate::error::TuneError::Validation(format!(
                    "LoRA layer {layer_idx} module '{module}': A buffer length {} does not \
                     match rank*d_in={expected_a} (rank={}, d_in={})",
                    layer.a.len(),
                    layer.rank,
                    layer.d_in
                )));
            }
            let expected_b = layer.d_out.checked_mul(layer.rank).ok_or_else(|| {
                crate::error::TuneError::Validation(format!(
                    "LoRA layer {layer_idx} module '{module}': d_out*rank overflowed usize \
                     (d_out={}, rank={})",
                    layer.d_out, layer.rank
                ))
            })?;
            if layer.b.len() != expected_b {
                return Err(crate::error::TuneError::Validation(format!(
                    "LoRA layer {layer_idx} module '{module}': B buffer length {} does not \
                     match d_out*rank={expected_b} (d_out={}, rank={})",
                    layer.b.len(),
                    layer.d_out,
                    layer.rank
                )));
            }
        }
        Ok(Self { config, layers })
    }

    /// Return the validated adapter configuration.
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Return the adapter weights keyed by transformer layer and module name.
    pub fn layers(&self) -> &HashMap<(usize, String), LoraLayer> {
        &self.layers
    }

    fn layers_mut(&mut self) -> &mut HashMap<(usize, String), LoraLayer> {
        &mut self.layers
    }

    /// Add this adapter's correction to one projection output in place.
    /// A missing `(layer_idx, module)` layer is a no-op; slices must match its shape.
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#adapter-representation-and-inference) for the matrix layout.
    ///
    /// This is called once per hooked row (see
    /// `lattice_inference::lora_hook::apply_lora_rows`), so the lookup
    /// below scans the layer map with a borrowed `&str` instead of hashing
    /// an owned `(usize, String)` key — allocating a `String` per row here
    /// would dominate the hot path long before the scan itself could. The
    /// map is bounded by `num_layers * target_modules.len()`, so the scan
    /// stays cheap.
    pub fn apply(&self, layer_idx: usize, module: &str, x: &[f32], base_output: &mut [f32]) {
        let lora_layer = self
            .layers()
            .iter()
            .find(|((idx, m), _)| *idx == layer_idx && m == module)
            .map(|(_, layer)| layer);
        if let Some(lora_layer) = lora_layer {
            let scale = self.config().scale();
            apply_lora(lora_layer, scale, x, base_output);
        }
    }

    /// Check if the adapter has weights for a specific layer and module.
    pub fn has_adapter(&self, layer_idx: usize, module: &str) -> bool {
        self.layers().contains_key(&(layer_idx, module.to_string()))
    }

    /// Return the number of adapted projection layers.
    pub fn num_adapted_layers(&self) -> usize {
        self.layers().len()
    }

    /// Return the total number of LoRA parameters (A + B matrices).
    pub fn num_parameters(&self) -> usize {
        self.layers().values().map(|l| l.a.len() + l.b.len()).sum()
    }

    /// Return `(layer_idx, module_name)` pairs whose module is not in `known`.
    ///
    /// Useful for detecting typos in adapter target modules before inference.
    pub fn validate_modules(&self, known: &[&str]) -> Vec<(usize, String)> {
        self.layers()
            .keys()
            .filter(|(_, m)| !known.iter().any(|k| k == m))
            .cloned()
            .collect()
    }
}

#[cfg(feature = "inference-hook")]
impl LoraAdapter {
    /// Validate adapter dimensions against a Qwen3.5 model configuration.
    ///
    /// Returns the first invalid layer, module, or projection-shape mismatch.
    /// Call it after loading and before
    /// [`set_lora`](lattice_inference::model::qwen35::Qwen35Model::set_lora).
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#adapter-validation) for projection shape rules.
    ///
    /// Per-layer `a`/`b` buffer lengths are not re-checked here: [`Self::new`]
    /// already rejects any layer whose buffers don't match its own
    /// `rank`/`d_in`/`d_out`, and it's the only way to construct a
    /// `LoraAdapter`, so every instance already satisfies that invariant.
    pub fn validate_against(
        &self,
        config: &lattice_inference::model::qwen35_config::Qwen35Config,
    ) -> crate::error::Result<()> {
        self.config().validate()?;
        for ((layer_idx, module), layer) in self.layers() {
            if *layer_idx >= config.num_hidden_layers {
                return Err(crate::error::TuneError::Validation(format!(
                    "LoRA layer index {layer_idx} >= model num_hidden_layers {} (module: {module})",
                    config.num_hidden_layers
                )));
            }

            let is_full = config.is_full_attention(*layer_idx);
            let (expected_d_in, expected_d_out) = match (module.as_str(), is_full) {
                ("q_proj", true) => (config.hidden_size, 2 * config.full_q_dim()),
                ("k_proj", true) => (config.hidden_size, config.full_kv_dim()),
                ("v_proj", true) => (config.hidden_size, config.full_kv_dim()),
                ("o_proj", true) => (config.full_q_dim(), config.hidden_size),
                ("in_proj_qkv", false) => (config.hidden_size, config.linear_qkv_dim()),
                ("in_proj_z", false) => (config.hidden_size, config.linear_output_dim()),
                // GDN beta/alpha widths are per value head, never key head.
                ("in_proj_b", false) => (config.hidden_size, config.linear_num_value_heads()),
                ("in_proj_a", false) => (config.hidden_size, config.linear_num_value_heads()),
                ("out_proj", false) => (config.linear_output_dim(), config.hidden_size),
                ("gate_proj", _) => (config.hidden_size, config.intermediate_size),
                ("up_proj", _) => (config.hidden_size, config.intermediate_size),
                ("down_proj", _) => (config.intermediate_size, config.hidden_size),
                (m, _) => {
                    return Err(crate::error::TuneError::Validation(format!(
                        "LoRA module '{m}' (layer {layer_idx}) is not a recognised Qwen3.5 projection"
                    )));
                }
            };

            if layer.d_in != expected_d_in || layer.d_out != expected_d_out {
                return Err(crate::error::TuneError::Validation(format!(
                    "LoRA adapter dims mismatch for layer {layer_idx} module '{module}': \
                     adapter has (d_in={}, d_out={}) but model expects (d_in={expected_d_in}, d_out={expected_d_out})",
                    layer.d_in, layer.d_out
                )));
            }
        }
        Ok(())
    }

    /// Validate adapter dimensions against a BERT cross-encoder model's
    /// geometry.
    ///
    /// Returns the first invalid layer, module, or projection-shape
    /// mismatch. Call it before hooked BERT scoring (see
    /// [`lattice_inference::model::cross_encoder::CrossEncoderModel::score_with_hook`],
    /// which validates through the [`LoraHook`](lattice_inference::lora_hook::LoraHook)
    /// trait object below) — this is the BERT counterpart to
    /// [`Self::validate_against`], which covers Qwen3.5 module shapes.
    ///
    /// Per-layer `a`/`b` buffer lengths are not re-checked here for the same
    /// reason as `validate_against`: [`Self::new`] already rejects any layer
    /// whose buffers don't match its own `rank`/`d_in`/`d_out`.
    pub fn validate_against_bert(
        &self,
        num_hidden_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> crate::error::Result<()> {
        self.config().validate()?;
        for ((layer_idx, module), layer) in self.layers() {
            if *layer_idx >= num_hidden_layers {
                return Err(crate::error::TuneError::Validation(format!(
                    "LoRA layer index {layer_idx} >= BERT num_hidden_layers {num_hidden_layers} (module: {module})"
                )));
            }

            let (expected_d_in, expected_d_out) = match module.as_str() {
                "query" | "key" | "value" | "attn_output" => (hidden_size, hidden_size),
                "ffn_intermediate" => (hidden_size, intermediate_size),
                "ffn_output" => (intermediate_size, hidden_size),
                m => {
                    return Err(crate::error::TuneError::Validation(format!(
                        "LoRA module '{m}' (layer {layer_idx}) is not a recognised BERT cross-encoder projection"
                    )));
                }
            };

            if layer.d_in != expected_d_in || layer.d_out != expected_d_out {
                return Err(crate::error::TuneError::Validation(format!(
                    "LoRA adapter dims mismatch for layer {layer_idx} module '{module}': \
                     adapter has (d_in={}, d_out={}) but BERT model expects (d_in={expected_d_in}, d_out={expected_d_out})",
                    layer.d_in, layer.d_out
                )));
            }
        }
        Ok(())
    }
}

// Delegate inference hooks to the adapter's application path.
#[cfg(feature = "inference-hook")]
impl lattice_inference::lora_hook::LoraHook for LoraAdapter {
    fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
        // Delegate to the existing apply method
        LoraAdapter::apply(self, layer_idx, module, x, output);
    }

    fn validate_against(
        &self,
        config: &lattice_inference::model::qwen35_config::Qwen35Config,
    ) -> Result<(), String> {
        LoraAdapter::validate_against(self, config).map_err(|e| e.to_string())
    }

    fn validate_against_bert(
        &self,
        num_hidden_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<(), String> {
        LoraAdapter::validate_against_bert(self, num_hidden_layers, hidden_size, intermediate_size)
            .map_err(|e| e.to_string())
    }

    fn is_active(&self, layer_idx: usize, module: &str) -> bool {
        LoraAdapter::has_adapter(self, layer_idx, module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_adapter() -> LoraAdapter {
        let config = LoraConfig {
            rank: 2,
            alpha: 4.0, // scale = 4.0 / 2 = 2.0
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        };

        let mut layers = HashMap::new();

        // Layer 0, q_proj: rank=2, d_in=4, d_out=4
        // A = identity-like (first 2 components), B = scale
        layers.insert(
            (0, "q_proj".into()),
            LoraLayer {
                a: vec![
                    1.0, 0.0, 0.0, 0.0, // row 0
                    0.0, 1.0, 0.0, 0.0, // row 1
                ],
                b: vec![
                    1.0, 0.0, // row 0
                    0.0, 1.0, // row 1
                    0.0, 0.0, // row 2
                    0.0, 0.0, // row 3
                ],
                d_in: 4,
                d_out: 4,
                rank: 2,
            },
        );

        LoraAdapter::new(config, layers).expect("valid adapter config")
    }

    #[test]
    fn test_config_scale() {
        let config = LoraConfig {
            rank: 8,
            alpha: 16.0,
            target_modules: vec![],
        };
        assert!((config.scale() - 2.0).abs() < 1e-6);

        let config_zero = LoraConfig {
            rank: 0,
            alpha: 1.0,
            target_modules: vec![],
        };
        assert_eq!(config_zero.scale(), 0.0);
    }

    #[test]
    fn test_config_rejects_non_finite_alpha_and_scale_fails_closed() {
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let config = LoraConfig {
                rank: 8,
                alpha,
                target_modules: vec![],
            };

            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("alpha must be finite"));
            assert_eq!(config.scale(), 0.0);
        }
    }

    #[test]
    fn test_adapter_construction_rejects_non_finite_alpha() {
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result = LoraAdapter::new(
                LoraConfig {
                    rank: 8,
                    alpha,
                    target_modules: vec![],
                },
                HashMap::new(),
            );

            let err = result.expect_err("non-finite alpha must reject adapter construction");
            assert!(err.to_string().contains("alpha must be finite"));
        }
    }

    #[test]
    fn test_adapter_apply() {
        let adapter = make_test_adapter();

        // x = [1, 2, 3, 4], base = [0, 0, 0, 0]
        // A @ x = [1, 2] (picks first 2 components)
        // B @ [1, 2] = [1, 2, 0, 0]
        // scale = 2.0
        // output = [0, 0, 0, 0] + 2.0 * [1, 2, 0, 0] = [2, 4, 0, 0]
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        adapter.apply(0, "q_proj", &x, &mut output);

        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0).abs() < 1e-6);
        assert!((output[2] - 0.0).abs() < 1e-6);
        assert!((output[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapter_noop_for_missing_module() {
        let adapter = make_test_adapter();

        let x = [1.0, 2.0, 3.0, 4.0];
        let mut output = [10.0, 20.0, 30.0, 40.0];
        // v_proj at layer 0 has no adapter -> should be a no-op
        adapter.apply(0, "v_proj", &x, &mut output);

        assert!((output[0] - 10.0).abs() < 1e-6);
        assert!((output[1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapter_noop_for_wrong_layer() {
        let adapter = make_test_adapter();

        let x = [1.0, 2.0, 3.0, 4.0];
        let mut output = [10.0, 20.0, 30.0, 40.0];
        // layer 1 has no adapter -> should be a no-op
        adapter.apply(1, "q_proj", &x, &mut output);

        assert!((output[0] - 10.0).abs() < 1e-6);
    }

    /// Regression for the per-row BERT LoRA dispatch path
    /// (`lattice_inference::lora_hook::apply_lora_rows` calls `apply()` once
    /// per token row): each row's output must depend only on that row's own
    /// input, independent of how many rows came before it. This pins the
    /// same per-row output the hash-keyed lookup produced before it was
    /// replaced with a borrowed-key scan.
    #[test]
    fn test_adapter_apply_over_multiple_token_rows_matches_per_row_reference() {
        let adapter = make_test_adapter();

        let rows: [[f32; 4]; 3] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.0; 4]];

        for row in rows {
            let mut output = [0.0f32; 4];
            adapter.apply(0, "q_proj", &row, &mut output);

            // Same reference as `test_adapter_apply`: A picks the first two
            // components of `x`, B is the 4x2 permutation built in
            // `make_test_adapter`, scale = 2.0.
            let expected = [2.0 * row[0], 2.0 * row[1], 0.0, 0.0];
            for (got, want) in output.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-6,
                    "got {output:?}, want {expected:?}"
                );
            }
        }
    }

    #[test]
    fn test_has_adapter() {
        let adapter = make_test_adapter();
        assert!(adapter.has_adapter(0, "q_proj"));
        assert!(!adapter.has_adapter(0, "v_proj"));
        assert!(!adapter.has_adapter(1, "q_proj"));
    }

    #[test]
    fn test_num_parameters() {
        let adapter = make_test_adapter();
        // A: 2*4 = 8, B: 4*2 = 8 => total 16
        assert_eq!(adapter.num_parameters(), 16);
    }

    #[test]
    fn test_num_adapted_layers() {
        let adapter = make_test_adapter();
        assert_eq!(adapter.num_adapted_layers(), 1);
    }

    #[test]
    fn test_validate_modules_all_known() {
        let adapter = make_test_adapter();
        let unknown = adapter.validate_modules(&["q_proj", "v_proj", "k_proj"]);
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_validate_modules_typo() {
        let config = LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["q_porj".into()],
        };
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_porj".into()),
            LoraLayer {
                a: vec![1.0; 8],
                b: vec![1.0; 8],
                d_in: 4,
                d_out: 4,
                rank: 2,
            },
        );
        let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");
        let unknown = adapter.validate_modules(&["q_proj", "v_proj"]);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0], (0, "q_porj".to_string()));
    }

    #[test]
    fn test_validate_modules_empty_adapter() {
        let config = LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec![],
        };
        let adapter = LoraAdapter::new(config, HashMap::new()).expect("valid adapter config");
        let unknown = adapter.validate_modules(&["q_proj"]);
        assert!(unknown.is_empty());
    }

    /// Regression for #972: a layer with *correct* `d_in`/`d_out` (what
    /// `validate_against` checked before this fix) but a short `a` buffer
    /// must still be rejected at construction, not admitted to later panic
    /// (slice-out-of-bounds) inside `apply_lora`.
    #[test]
    fn test_new_rejects_a_buffer_shorter_than_rank_times_d_in() {
        let config = LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["q_proj".into()],
        };
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".into()),
            LoraLayer {
                a: vec![0.0; 4 * 8 - 1], // one element short of rank(4) * d_in(8)
                b: vec![0.0; 8 * 4],
                d_in: 8,
                d_out: 8,
                rank: 4,
            },
        );
        let err = LoraAdapter::new(config, layers)
            .expect_err("A buffer shorter than rank*d_in must be rejected");
        assert!(err.to_string().contains("A buffer length"));
    }

    /// Same as above for `b`, the other operand `apply_lora` slices by rank.
    #[test]
    fn test_new_rejects_b_buffer_longer_than_d_out_times_rank() {
        let config = LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["q_proj".into()],
        };
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".into()),
            LoraLayer {
                a: vec![0.0; 4 * 8],
                b: vec![0.0; 8 * 4 + 3], // padded beyond d_out(8) * rank(4)
                d_in: 8,
                d_out: 8,
                rank: 4,
            },
        );
        let err = LoraAdapter::new(config, layers)
            .expect_err("B buffer longer than d_out*rank must be rejected");
        assert!(err.to_string().contains("B buffer length"));
    }

    #[cfg(feature = "inference-hook")]
    mod validate_against_tests {
        use super::*;
        use lattice_inference::model::qwen35_config::Qwen35Config;

        fn make_adapter_for_layer(
            layer_idx: usize,
            module: &str,
            d_in: usize,
            d_out: usize,
        ) -> LoraAdapter {
            let rank = 4;
            let mut layers = HashMap::new();
            layers.insert(
                (layer_idx, module.to_string()),
                LoraLayer {
                    a: vec![0.0; rank * d_in],
                    b: vec![0.0; d_out * rank],
                    d_in,
                    d_out,
                    rank,
                },
            );
            LoraAdapter::new(
                LoraConfig {
                    rank,
                    alpha: rank as f32,
                    target_modules: vec![module.to_string()],
                },
                layers,
            )
            .expect("valid adapter config")
        }

        #[test]
        fn test_validate_against_layer_out_of_bounds() {
            let cfg = Qwen35Config::qwen35_0_8b();
            // layer 999 does not exist
            let adapter = make_adapter_for_layer(999, "q_proj", 1024, 4096);
            assert!(adapter.validate_against(&cfg).is_err());
        }

        #[test]
        fn test_validate_against_dim_mismatch() {
            let cfg = Qwen35Config::qwen35_0_8b();
            // 0.8b: hidden=1024, full_q_dim=8*256=2048 → q_proj expects (1024, 4096)
            // Supply 2b dims (hidden=2048, full_q_dim=4096 → d_out=8192) — wrong for 0.8b.
            // Layer 3 is full-attention in the 24-layer 0.8b config.
            let adapter = make_adapter_for_layer(3, "q_proj", 2048, 8192);
            assert!(adapter.validate_against(&cfg).is_err());
        }

        #[test]
        fn test_validate_against_correct_dims_passes() {
            let cfg = Qwen35Config::qwen35_0_8b();
            // Layer 3 is full-attention; q_proj: d_in=hidden=1024, d_out=2*full_q_dim=4096.
            let adapter = make_adapter_for_layer(3, "q_proj", 1024, 4096);
            assert!(adapter.validate_against(&cfg).is_ok());
        }

        #[test]
        fn test_validate_against_mlp_correct() {
            let cfg = Qwen35Config::qwen35_0_8b();
            // gate_proj on any layer: d_in=hidden=1024, d_out=intermediate=3584.
            let adapter = make_adapter_for_layer(0, "gate_proj", 1024, 3584);
            assert!(adapter.validate_against(&cfg).is_ok());
        }

        #[test]
        fn test_validate_against_unknown_module_errors() {
            let cfg = Qwen35Config::qwen35_0_8b();
            let adapter = make_adapter_for_layer(3, "xq_proj_typo", 1024, 4096);
            let err = adapter.validate_against(&cfg).unwrap_err();
            assert!(err.to_string().contains("not a recognised"));
        }

        #[test]
        fn test_validate_against_gdn_all_modules_pass() {
            // Regression: train_grad_full --save emits all five GDN LoRA modules
            // (in_proj_qkv/z/b/a, out_proj). The forward applies every one of them
            // (gdn_fused.rs), so validate_against must accept each with the dims the
            // loader and trainer use. Dims are derived from the config, not hardcoded,
            // so this stays correct if the reference dims change.
            let cfg = Qwen35Config::qwen35_0_8b();
            let gdn_layer = (0..cfg.num_hidden_layers)
                .find(|&i| !cfg.is_full_attention(i))
                .expect("0.8b config has linear-attention layers");
            let h = cfg.hidden_size;
            let cases = [
                ("in_proj_qkv", h, cfg.linear_qkv_dim()),
                ("in_proj_z", h, cfg.linear_output_dim()),
                ("in_proj_b", h, cfg.linear_num_value_heads()),
                ("in_proj_a", h, cfg.linear_num_value_heads()),
                ("out_proj", cfg.linear_output_dim(), h),
            ];
            for (module, d_in, d_out) in cases {
                let adapter = make_adapter_for_layer(gdn_layer, module, d_in, d_out);
                assert!(
                    adapter.validate_against(&cfg).is_ok(),
                    "GDN LoRA module {module} should validate (d_in={d_in}, d_out={d_out})"
                );
            }
        }

        /// Regression for #792: in_proj_b/in_proj_a
        /// dims must be keyed by `linear_num_value_heads()`, not
        /// `linear_num_key_heads`. Asymmetric-head configs (key_heads !=
        /// value_heads) are the only ones that can tell these apart —
        /// `qwen35_0_8b` has key_heads == value_heads == 16 and would pass
        /// either way.
        #[test]
        fn test_validate_against_gdn_asymmetric_value_heads_35b_a3b() {
            let cfg = Qwen35Config::qwen36_35b_a3b();
            assert_eq!(cfg.linear_num_key_heads, 16);
            assert_eq!(cfg.linear_num_value_heads(), 32);
            let gdn_layer = (0..cfg.num_hidden_layers)
                .find(|&i| !cfg.is_full_attention(i))
                .expect("config has linear-attention layers");
            let h = cfg.hidden_size;

            // Correct dims (value_heads=32) must validate.
            let ok_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 32);
            assert!(ok_b.validate_against(&cfg).is_ok());
            let ok_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 32);
            assert!(ok_a.validate_against(&cfg).is_ok());

            // The old (wrong) key-head dim (16) must be rejected.
            let bad_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 16);
            assert!(
                bad_b.validate_against(&cfg).is_err(),
                "in_proj_b with key-head dim (16) must fail on a 16-key/32-value config"
            );
            let bad_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 16);
            assert!(
                bad_a.validate_against(&cfg).is_err(),
                "in_proj_a with key-head dim (16) must fail on a 16-key/32-value config"
            );
        }

        #[test]
        fn test_validate_against_gdn_asymmetric_value_heads_27b() {
            let cfg = Qwen35Config::qwen36_27b();
            assert_eq!(cfg.linear_num_key_heads, 16);
            assert_eq!(cfg.linear_num_value_heads(), 48);
            let gdn_layer = (0..cfg.num_hidden_layers)
                .find(|&i| !cfg.is_full_attention(i))
                .expect("config has linear-attention layers");
            let h = cfg.hidden_size;

            // Correct dims (value_heads=48) must validate.
            let ok_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 48);
            assert!(ok_b.validate_against(&cfg).is_ok());
            let ok_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 48);
            assert!(ok_a.validate_against(&cfg).is_ok());

            // The old (wrong) key-head dim (16) must be rejected.
            let bad_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 16);
            assert!(
                bad_b.validate_against(&cfg).is_err(),
                "in_proj_b with key-head dim (16) must fail on a 16-key/48-value config"
            );
            let bad_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 16);
            assert!(
                bad_a.validate_against(&cfg).is_err(),
                "in_proj_a with key-head dim (16) must fail on a 16-key/48-value config"
            );
        }

        /// Regression for #753: `Qwen35Model::set_lora` rejects a mismatched
        /// adapter by calling through `dyn LoraHook::validate_against`, not
        /// the inherent `LoraAdapter::validate_against`. This confirms the
        /// trait-object glue (`impl LoraHook for LoraAdapter`) actually
        /// delegates to the inherent method instead of defaulting to the
        /// trait's no-op `Ok(())`, for both a mismatched and a matching
        /// adapter.
        #[test]
        fn test_lora_hook_trait_validate_against_delegates_to_inherent_method() {
            use lattice_inference::lora_hook::LoraHook;

            let cfg = Qwen35Config::qwen35_0_8b();

            // Same mismatch as `test_validate_against_dim_mismatch`: 2b dims
            // supplied against the 0.8b config's expected q_proj shape.
            // Fully-qualified syntax forces dispatch through the trait method
            // (not `LoraAdapter`'s own inherent `validate_against`, which dot
            // syntax would otherwise prefer).
            let mismatched = make_adapter_for_layer(3, "q_proj", 2048, 8192);
            assert!(
                LoraHook::validate_against(&mismatched, &cfg).is_err(),
                "the LoraHook trait method must surface the same dim mismatch as \
                 the inherent LoraAdapter::validate_against"
            );

            let matching = make_adapter_for_layer(3, "q_proj", 1024, 4096);
            assert!(
                LoraHook::validate_against(&matching, &cfg).is_ok(),
                "the LoraHook trait method must accept an adapter with correct dims"
            );
        }

        /// Regression for #972: a layer whose
        /// projection dims exactly match a real Qwen3.5 model's `q_proj`
        /// (the only thing `validate_against` checked before this fix) but
        /// whose `a` buffer is short must be rejected before it can ever
        /// reach `validate_against` (and hence `Qwen35Model::set_lora`,
        /// which calls it — see `test_lora_hook_trait_validate_against_delegates_to_inherent_method`
        /// above). `LoraAdapter::new` is the sole construction path, so
        /// rejecting it there is equivalent to rejecting it at the
        /// `set_lora` gate: the malformed adapter never becomes an
        /// installable `LoraAdapter` value at all.
        #[test]
        fn test_new_rejects_malformed_buffer_matching_real_model_projection_dims() {
            // Layer 3 is full-attention in `Qwen35Config::qwen35_0_8b()`;
            // q_proj there expects d_in=hidden=1024, d_out=2*full_q_dim=4096
            // (see `test_validate_against_correct_dims_passes` above).
            let (layer_idx, module, d_in, d_out) = (3usize, "q_proj", 1024usize, 4096usize);
            let rank = 4;

            let config = LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec![module.to_string()],
            };
            let mut layers = HashMap::new();
            layers.insert(
                (layer_idx, module.to_string()),
                LoraLayer {
                    a: vec![0.0; rank * d_in - 1], // one short of rank*d_in
                    b: vec![0.0; d_out * rank],
                    d_in,
                    d_out,
                    rank,
                },
            );

            let err = LoraAdapter::new(config, layers).expect_err(
                "a malformed A buffer must be rejected at construction, even when \
                 d_in/d_out exactly match a real model's projection shape",
            );
            assert!(err.to_string().contains("A buffer length"));

            // No `LoraAdapter` value exists to pass to `validate_against` or
            // `set_lora` — construction itself is the gate here.
        }
    }

    /// BERT counterpart to `validate_against_tests`: covers
    /// `LoraAdapter::validate_against_bert`, the geometry check
    /// `CrossEncoderModel::score_with_hook` calls before hooked scoring
    /// (lattice#1031 follow-up).
    #[cfg(feature = "inference-hook")]
    mod validate_against_bert_tests {
        use super::*;

        const NUM_HIDDEN_LAYERS: usize = 12;
        const HIDDEN_SIZE: usize = 384;
        const INTERMEDIATE_SIZE: usize = 1536;

        fn make_bert_adapter(module: &str, d_in: usize, d_out: usize) -> LoraAdapter {
            let rank = 4;
            let mut layers = HashMap::new();
            layers.insert(
                (0, module.to_string()),
                LoraLayer {
                    a: vec![0.0; rank * d_in],
                    b: vec![0.0; d_out * rank],
                    d_in,
                    d_out,
                    rank,
                },
            );
            LoraAdapter::new(
                LoraConfig {
                    rank,
                    alpha: rank as f32,
                    target_modules: vec![module.to_string()],
                },
                layers,
            )
            .expect("valid adapter config")
        }

        #[test]
        fn test_validate_against_bert_oversized_d_out_rejected() {
            // The original bug: a self-consistent adapter declaring d_out >
            // hidden_size, which `apply_lora` would slice `output[..d_out]`
            // out of bounds on past a debug_assert release builds compile out.
            let adapter = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE + 1);
            let err = adapter
                .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                .expect_err("d_out > hidden_size must be rejected");
            assert!(err.to_string().contains("dims mismatch"));
        }

        #[test]
        fn test_validate_against_bert_mismatched_d_in_rejected() {
            // Even when d_out happens to fit, a wrong d_in is a
            // silent-wrong-math bug, not merely a panic risk.
            let adapter = make_bert_adapter("query", HIDDEN_SIZE + 1, HIDDEN_SIZE);
            let err = adapter
                .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                .expect_err("mismatched d_in must be rejected");
            assert!(err.to_string().contains("dims mismatch"));
        }

        #[test]
        fn test_validate_against_bert_layer_out_of_bounds_rejected() {
            let adapter = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE);
            let err = adapter
                .validate_against_bert(0, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                .expect_err("layer 0 >= num_hidden_layers 0 must be rejected");
            assert!(err.to_string().contains("num_hidden_layers"));
        }

        #[test]
        fn test_validate_against_bert_unknown_module_rejected() {
            let adapter = make_bert_adapter("xquery_typo", HIDDEN_SIZE, HIDDEN_SIZE);
            let err = adapter
                .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                .expect_err("unrecognised BERT module must be rejected");
            assert!(err.to_string().contains("not a recognised"));
        }

        #[test]
        fn test_validate_against_bert_all_modules_correct_dims_pass() {
            let cases = [
                ("query", HIDDEN_SIZE, HIDDEN_SIZE),
                ("key", HIDDEN_SIZE, HIDDEN_SIZE),
                ("value", HIDDEN_SIZE, HIDDEN_SIZE),
                ("attn_output", HIDDEN_SIZE, HIDDEN_SIZE),
                ("ffn_intermediate", HIDDEN_SIZE, INTERMEDIATE_SIZE),
                ("ffn_output", INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ];
            for (module, d_in, d_out) in cases {
                let adapter = make_bert_adapter(module, d_in, d_out);
                assert!(
                    adapter
                        .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                        .is_ok(),
                    "module {module} with correct dims (d_in={d_in}, d_out={d_out}) should validate"
                );
            }
        }

        #[test]
        fn test_lora_hook_trait_validate_against_bert_delegates_to_inherent_method() {
            use lattice_inference::lora_hook::LoraHook;

            let mismatched = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE + 1);
            assert!(
                LoraHook::validate_against_bert(
                    &mismatched,
                    NUM_HIDDEN_LAYERS,
                    HIDDEN_SIZE,
                    INTERMEDIATE_SIZE
                )
                .is_err(),
                "the LoraHook trait method must surface the same dim mismatch as \
                 the inherent LoraAdapter::validate_against_bert"
            );

            let matching = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE);
            assert!(
                LoraHook::validate_against_bert(
                    &matching,
                    NUM_HIDDEN_LAYERS,
                    HIDDEN_SIZE,
                    INTERMEDIATE_SIZE
                )
                .is_ok(),
                "the LoraHook trait method must accept an adapter with correct dims"
            );
        }

        /// `LoraHook::is_active` must reflect whether this adapter actually
        /// has a layer for `(layer_idx, module)`, so `apply_lora_rows` can
        /// skip its per-row loop entirely for projections this adapter
        /// doesn't touch, instead of paying one no-op virtual call per row.
        #[test]
        fn test_lora_hook_trait_is_active_reflects_has_adapter() {
            use lattice_inference::lora_hook::LoraHook;

            let adapter = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE);
            assert!(
                LoraHook::is_active(&adapter, 0, "query"),
                "adapter has a layer for (0, query)"
            );
            assert!(
                !LoraHook::is_active(&adapter, 0, "key"),
                "adapter has no layer for (0, key)"
            );
            assert!(
                !LoraHook::is_active(&adapter, 5, "query"),
                "adapter has no layer for (5, query)"
            );
        }

        /// A layer with an empty `a` factor but a non-empty `b` factor and
        /// declared dims that match a real BERT projection's geometry is not
        /// a valid "untrained placeholder" (that state requires BOTH `a` and
        /// `b` empty) — it must be rejected at construction rather than
        /// reaching `apply_lora`, which indexes into `a` assuming it holds
        /// `rank * d_in` elements.
        #[test]
        fn test_new_rejects_empty_a_factor_with_matching_declared_dims() {
            let rank = 4;
            let mut layers = HashMap::new();
            layers.insert(
                (0, "query".to_string()),
                LoraLayer {
                    a: vec![],
                    b: vec![0.0; HIDDEN_SIZE * rank],
                    d_in: HIDDEN_SIZE,
                    d_out: HIDDEN_SIZE,
                    rank,
                },
            );
            let config = LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["query".to_string()],
            };
            let err = LoraAdapter::new(config, layers)
                .expect_err("an empty A factor with a populated B factor must be rejected");
            assert!(err.to_string().contains("A buffer length"));
        }

        /// Mirror of the above for an empty `b` factor with a populated `a`.
        #[test]
        fn test_new_rejects_empty_b_factor_with_matching_declared_dims() {
            let rank = 4;
            let mut layers = HashMap::new();
            layers.insert(
                (0, "query".to_string()),
                LoraLayer {
                    a: vec![0.0; rank * HIDDEN_SIZE],
                    b: vec![],
                    d_in: HIDDEN_SIZE,
                    d_out: HIDDEN_SIZE,
                    rank,
                },
            );
            let config = LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["query".to_string()],
            };
            let err = LoraAdapter::new(config, layers)
                .expect_err("an empty B factor with a populated A factor must be rejected");
            assert!(err.to_string().contains("B buffer length"));
        }

        /// A layer with BOTH `a` and `b` empty is the legitimate
        /// untrained-placeholder state (mirrors `save_peft_safetensors`,
        /// which skips these layers) and must still construct successfully.
        #[test]
        fn test_new_accepts_fully_empty_placeholder_layer() {
            let rank = 4;
            let mut layers = HashMap::new();
            layers.insert(
                (0, "query".to_string()),
                LoraLayer {
                    a: vec![],
                    b: vec![],
                    d_in: HIDDEN_SIZE,
                    d_out: HIDDEN_SIZE,
                    rank,
                },
            );
            let config = LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["query".to_string()],
            };
            assert!(
                LoraAdapter::new(config, layers).is_ok(),
                "a placeholder layer with both factors empty must still construct"
            );
        }
    }
}
