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
#[cfg(feature = "safetensors")]
mod safetensors;

pub use apply::apply_lora;

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
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
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

    /// Construct an adapter from pre-built components (for testing or
    /// when loading from a custom format).
    pub fn new(config: LoraConfig, layers: HashMap<(usize, String), LoraLayer>) -> Self {
        Self { config, layers }
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

        LoraAdapter::new(config, layers)
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
}
