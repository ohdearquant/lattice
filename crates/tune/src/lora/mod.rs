//! LoRA (Low-Rank Adaptation) adapter loading and inference integration.
//!
//! This module implements the Rust side of a LoRA fine-tuning pipeline:
//! **train (Python/PEFT) -> export safetensors -> load in Rust -> apply during inference**.
//!
//! LoRA decomposes weight updates into low-rank matrices:
//! `dW = B @ A` where A is `(rank, d_in)` and B is `(d_out, rank)`.
//! During inference, the output is modified as:
//! `output += (alpha / rank) * B @ (A @ x)`
//!
//! # Supported target modules
//!
//! For Qwen3.5-2B (and similar transformer architectures):
//! - **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
//! - **MLP**: `gate_proj`, `up_proj`, `down_proj`
//!
//! For BERT/CrossEncoder models:
//! - **Attention**: `query`, `key`, `value`, `attn_output`
//! - **FFN**: `ffn_intermediate`, `ffn_output`
//!
//! # Example
//!
//! ```ignore
//! use lattice_tune::lora::LoraAdapter;
//! use std::path::Path;
//!
//! // Load a PEFT-exported adapter
//! let adapter = LoraAdapter::from_safetensors(Path::new("adapter.safetensors"))?;
//!
//! // Apply to a projection output during inference
//! let x = vec![0.1f32; 2048];         // input activation
//! let mut output = vec![0.0f32; 2048]; // base projection output
//! adapter.apply(5, "q_proj", &x, &mut output);
//! ```

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

/// A single LoRA low-rank decomposition for one linear projection.
///
/// Stores the A and B matrices in row-major f32 layout:
/// - `a`: shape `(rank, d_in)` -- projects input down to rank
/// - `b`: shape `(d_out, rank)` -- projects rank up to output
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

/// A complete LoRA adapter: one [`LoraLayer`] per (layer_idx, module_name) pair.
///
/// Loaded from a PEFT-format safetensors file and applied at inference time
/// to modify the output of specific linear projections in the transformer.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Adapter configuration.
    pub config: LoraConfig,
    /// Per-layer, per-module LoRA weights.
    /// Key: `(layer_idx, module_name)` e.g. `(5, "q_proj")`.
    pub layers: HashMap<(usize, String), LoraLayer>,
}

impl LoraAdapter {
    /// Load a LoRA adapter from a PEFT-format safetensors file.
    ///
    /// Parses tensor keys like:
    /// `base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight`
    ///
    /// # Errors
    ///
    /// Returns an error if the file is invalid, has mismatched A/B pairs,
    /// or contains tensors with inconsistent ranks.
    #[cfg(feature = "safetensors")]
    pub fn from_safetensors(path: &Path) -> crate::error::Result<Self> {
        safetensors::load_peft_safetensors(path)
    }

    /// Save this LoRA adapter to a PEFT-format safetensors file.
    ///
    /// Pass `governance: Some(..)` to embed provenance fields (name, owner,
    /// base/tokenizer rev, dtype, approval status) into the safetensors
    /// metadata header; `None` preserves the pre-#610 behavior of writing
    /// only `rank`/`alpha`/`target_modules`.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor serialization fails or the file cannot be written.
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
    /// # Errors
    ///
    /// Returns an error when the adapter configuration is invalid.
    pub fn new(
        config: LoraConfig,
        layers: HashMap<(usize, String), LoraLayer>,
    ) -> crate::error::Result<Self> {
        config.validate()?;
        Ok(Self { config, layers })
    }

    /// Apply the LoRA adapter to a single projection output.
    ///
    /// Looks up the adapter for `(layer_idx, module)`. If no adapter exists
    /// for that combination, this is a no-op (base output unchanged).
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index (0-based)
    /// * `module` - Module name, e.g. `"q_proj"`, `"gate_proj"`
    /// * `x` - Input activation vector (length = d_in of the projection)
    /// * `base_output` - Base projection output to modify in-place (length = d_out)
    pub fn apply(&self, layer_idx: usize, module: &str, x: &[f32], base_output: &mut [f32]) {
        let key = (layer_idx, module.to_string());
        if let Some(lora_layer) = self.layers.get(&key) {
            let scale = self.config.scale();
            apply_lora(lora_layer, scale, x, base_output);
        }
    }

    /// Check if the adapter has weights for a specific layer and module.
    pub fn has_adapter(&self, layer_idx: usize, module: &str) -> bool {
        self.layers.contains_key(&(layer_idx, module.to_string()))
    }

    /// Return the number of adapted projection layers.
    pub fn num_adapted_layers(&self) -> usize {
        self.layers.len()
    }

    /// Return the total number of LoRA parameters (A + B matrices).
    pub fn num_parameters(&self) -> usize {
        self.layers.values().map(|l| l.a.len() + l.b.len()).sum()
    }

    /// Return `(layer_idx, module_name)` pairs whose module is not in `known`.
    ///
    /// Useful for detecting typos in adapter target modules before inference.
    pub fn validate_modules(&self, known: &[&str]) -> Vec<(usize, String)> {
        self.layers
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
    /// Checks every `(layer_idx, module)` pair in the adapter:
    /// - `layer_idx` is within `config.num_hidden_layers`
    /// - `d_in` / `d_out` match the projection dimensions the model expects
    ///
    /// Returns `Err(TuneError::Validation(...))` on the first mismatch found.
    /// Call this after loading and before
    /// [`set_lora`](lattice_inference::model::qwen35::Qwen35Model::set_lora)
    /// to surface dim errors before generation starts.
    pub fn validate_against(
        &self,
        config: &lattice_inference::model::qwen35_config::Qwen35Config,
    ) -> crate::error::Result<()> {
        self.config.validate()?;
        for ((layer_idx, module), layer) in &self.layers {
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
                // beta/alpha are projected per VALUE head (matches the shipping
                // gdn_fused forward and the f16 weight loader), not per key head
                // (#792: this was linear_num_key_heads, wrong for asymmetric
                // 4B/27B shapes where value_heads != key_heads).
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
}

// Implement LoraHook from lattice-inference so LoraAdapter can be injected
// into the inference forward pass.
#[cfg(feature = "inference-hook")]
impl lattice_inference::lora_hook::LoraHook for LoraAdapter {
    fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
        // Delegate to the existing apply method
        LoraAdapter::apply(self, layer_idx, module, x, output);
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
    }
}
