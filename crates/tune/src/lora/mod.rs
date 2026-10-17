//! LoRA adapters, lightweight training helpers, and inference integration.
//!
//! A layer stores row-major `A: (rank, d_in)` and `B: (d_out, rank)` and adds
//! `(alpha / rank) * B @ (A @ x)` to a base projection. The adapter constructor
//! rejects non-finite alpha values; a zero rank has an effective scale of zero.
//!
//! Module names: full-attention (GQA) `"q_proj"`, `"k_proj"`, `"v_proj"`, `"o_proj"`;
//! linear-attention (GDN) `"in_proj_qkv"`, `"in_proj_z"`, `"in_proj_b"`, `"in_proj_a"`,
//! `"out_proj"`; MLP `"gate_proj"`, `"up_proj"`, `"down_proj"`; BERT `"query"`, `"key"`,
//! `"value"`, `"attn_output"`, `"ffn_intermediate"`, `"ffn_output"`.
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
// Moved to `lattice_fann::training::router_update` (ADR-094 decision 3). Re-exported
// here so `lattice_tune::lora::router_update::*` keeps resolving for existing callers;
// that those callers compile unmodified is the move's acceptance test.
#[cfg(feature = "mixture")]
pub use lattice_fann::training::router_update;
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
    /// Tensor dtype label the adapter's in-memory `f32` buffers were
    /// converted from (e.g. `"f32"`, `"f16"`, `"bf16"`), matching
    /// [`lattice_fann::lora::LoraDescriptor::dtype`].
    pub dtype: String,
}

impl LoraConfig {
    /// Compute the LoRA scaling factor: `alpha / rank`.
    ///
    /// Delegates directly to [`lattice_fann::lora::effective_scale`], the
    /// single source of truth this crate shares with `lattice-inference`'s
    /// Metal LoRA path so the zero-rank and non-finite fallbacks cannot
    /// drift between the two. Deliberately bypasses [`Self::to_descriptor`]:
    /// that helper clones `target_modules` and `dtype` to build a full
    /// descriptor, and `scale()` is called once per matched row by
    /// [`LoraAdapter::apply`] — an allocation-free hot path that has no use
    /// for either field.
    pub fn scale(&self) -> f32 {
        lattice_fann::lora::effective_scale(self.rank, self.alpha)
    }

    /// Validate that the LoRA alpha and effective scale are finite.
    ///
    /// Deliberately does not check `target_modules` against any known-name
    /// list: this is the identity check [`LoraAdapter::new`] runs on every
    /// construction path (safetensors loading, blending, training), and the
    /// adapter representation is model-neutral (see the module docs) — a
    /// target module this crate does not yet recognize by name is not
    /// necessarily invalid, only unusable until something that knows the
    /// target architecture is asked to install it.
    /// [`LoraAdapter::new_with_descriptor`] and
    /// `LoraAdapter::validate_against`/`LoraAdapter::validate_against_bert`
    /// enforce a known-name check where the architecture is actually known;
    /// see `docs/lora-core.md#adapter-validation`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::TuneError::Validation`] when `alpha` or the
    /// effective `alpha / rank` scale is not finite.
    pub fn validate(&self) -> crate::error::Result<()> {
        self.to_descriptor()
            .validate()
            .map_err(crate::error::TuneError::Validation)
    }

    /// Convert to the shared cross-crate [`lattice_fann::lora::LoraDescriptor`].
    pub fn to_descriptor(&self) -> lattice_fann::lora::LoraDescriptor {
        lattice_fann::lora::LoraDescriptor {
            rank: self.rank,
            alpha: self.alpha,
            target_modules: self.target_modules.clone(),
            dtype: self.dtype.clone(),
        }
    }

    /// Build a `LoraConfig` from the shared cross-crate
    /// [`lattice_fann::lora::LoraDescriptor`].
    pub fn from_descriptor(descriptor: lattice_fann::lora::LoraDescriptor) -> Self {
        Self {
            rank: descriptor.rank,
            alpha: descriptor.alpha,
            target_modules: descriptor.target_modules,
            dtype: descriptor.dtype,
        }
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
    /// `validate_against` and `apply` — can rely
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

    /// Construct an adapter from a shared cross-crate
    /// [`lattice_fann::lora::LoraDescriptor`] and pre-built layers, enforcing
    /// that the descriptor's declared identity actually matches the layer
    /// data — the check [`Self::new`] deliberately does not make.
    ///
    /// [`Self::new`] only checks each layer's buffers against that *same*
    /// layer's own `rank`/`d_in`/`d_out`; it never compares `config.rank`
    /// (what [`LoraConfig::scale`] and hence [`Self::apply`] actually use)
    /// against the layers it is handed, and never compares `config.target_modules`
    /// against the layer keys present. That permissiveness is required by
    /// [`crate::lora::blend::blend_lora_adapters`], which legitimately builds
    /// adapters whose per-projection rank varies (the blended rank is the
    /// *sum* of the source ranks for that projection, which can differ across
    /// projection groups when the source adapters don't all cover the same
    /// modules) alongside a single representative `config.rank`. This
    /// constructor is for the opposite case: a single adapter loaded from a
    /// descriptor that is supposed to describe every layer uniformly (e.g. an
    /// external format's parsed identity plus its layers), where a
    /// disagreement between the two is a bug in the caller, not a legitimate
    /// blend artifact. It mirrors the identity check `lattice-inference`'s
    /// Metal `load_lora_adapter_with_descriptor` performs at its own entry
    /// point, so a descriptor's `rank`/`target_modules` mean the same thing
    /// on both sides of the tune/Metal boundary.
    ///
    /// [`Self::new`] remains the right choice for blend-style construction,
    /// or any caller that already owns the rank/module invariant itself;
    /// this constructor is the stricter alternative for callers that want it
    /// enforced instead of assumed.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor's alpha/effective scale is not
    /// finite ([`LoraConfig::validate`]), if `target_modules` contains a
    /// name outside [`lattice_fann::lora::KNOWN_LORA_TARGET_MODULES`], if
    /// any non-placeholder layer's `rank` disagrees with `descriptor.rank`,
    /// if the set of modules present in `layers` disagrees with
    /// `descriptor.target_modules`, or for any reason [`Self::new`] itself
    /// would reject the call.
    ///
    /// The known-name check lives here rather than in [`LoraConfig::validate`]
    /// (which [`Self::new`] also calls) so that the permissive bare
    /// constructor stays permissive; this mirrors
    /// `MetalQwen35State::load_lora_adapter_with_descriptor`, which performs
    /// the identical two-step (`validate` then `validate_target_modules`) at
    /// its own entry point.
    pub fn new_with_descriptor(
        descriptor: lattice_fann::lora::LoraDescriptor,
        layers: HashMap<(usize, String), LoraLayer>,
    ) -> crate::error::Result<Self> {
        let config = LoraConfig::from_descriptor(descriptor);
        config.validate()?;
        lattice_fann::lora::validate_target_modules(
            &config.target_modules,
            lattice_fann::lora::KNOWN_LORA_TARGET_MODULES,
        )
        .map_err(crate::error::TuneError::Validation)?;

        for ((layer_idx, module), layer) in &layers {
            // Mirrors `Self::new`'s own placeholder exemption: an
            // untrained/not-yet-populated module carries no rank-dependent
            // data for this check to disagree over.
            if layer.a.is_empty() && layer.b.is_empty() {
                continue;
            }
            if layer.rank != config.rank {
                return Err(crate::error::TuneError::Validation(format!(
                    "LoRA layer {layer_idx} module '{module}': layer rank {} disagrees with \
                     descriptor rank {}",
                    layer.rank, config.rank
                )));
            }
        }

        let mut layer_modules: Vec<&str> = layers.keys().map(|(_, m)| m.as_str()).collect();
        layer_modules.sort_unstable();
        layer_modules.dedup();
        let mut declared_modules: Vec<&str> =
            config.target_modules.iter().map(String::as_str).collect();
        declared_modules.sort_unstable();
        declared_modules.dedup();
        if layer_modules != declared_modules {
            return Err(crate::error::TuneError::Validation(format!(
                "LoRA adapter modules {layer_modules:?} disagree with descriptor \
                 target_modules {declared_modules:?}"
            )));
        }

        Self::new(config, layers)
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

    /// Look up the layer for `(layer_idx, module)`, if any.
    ///
    /// Called once per hooked row (see
    /// `lattice_inference::lora_hook::apply_lora_rows`), so this scans the
    /// layer map with a borrowed `&str` instead of hashing an owned
    /// `(usize, String)` key — allocating a `String` per row here would
    /// dominate the hot path long before the scan itself could. The map is
    /// bounded by `num_layers * target_modules.len()`, so the scan stays
    /// cheap. Both [`Self::apply`] and [`Self::has_adapter`] route through
    /// this one lookup so they can never disagree on what counts as present.
    fn find_layer(&self, layer_idx: usize, module: &str) -> Option<&LoraLayer> {
        self.layers()
            .iter()
            .find(|((idx, m), _)| *idx == layer_idx && m == module)
            .map(|(_, layer)| layer)
    }

    /// Add this adapter's correction to one projection output in place.
    /// A missing `(layer_idx, module)` layer is a no-op; slices must match its shape.
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#adapter-representation-and-inference) for the matrix layout.
    pub fn apply(&self, layer_idx: usize, module: &str, x: &[f32], base_output: &mut [f32]) {
        if let Some(lora_layer) = self.find_layer(layer_idx, module) {
            let scale = self.config().scale();
            apply_lora(lora_layer, scale, x, base_output);
        }
    }

    /// Check if the adapter has weights for a specific layer and module.
    ///
    /// An untrained placeholder layer (both `a` and `b` empty — see
    /// [`Self::new`]) is present in the map but reports `false` here: it
    /// carries no correction for `apply` to add, so callers that use this to
    /// skip per-row dispatch (e.g. `apply_lora_rows`) don't pay for a layer
    /// that would only ever no-op.
    pub fn has_adapter(&self, layer_idx: usize, module: &str) -> bool {
        self.find_layer(layer_idx, module)
            .is_some_and(|layer| !(layer.a.is_empty() && layer.b.is_empty()))
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

            let expected = lattice_inference::lora_hook::qwen35_projection_shape(
                config,
                *layer_idx,
                module,
            )
            .map_err(|_| {
                crate::error::TuneError::Validation(format!(
                    "LoRA module '{module}' (layer {layer_idx}) is not a recognised Qwen3.5 projection"
                ))
            })?;
            let expected_d_in = expected.d_in;
            let expected_d_out = expected.d_out;

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
    ///
    /// `rank` is deliberately not bounded here. A factorization with
    /// `rank > min(d_in, d_out)` is redundant but valid, and rejecting it
    /// would make a legitimately constructed adapter unscoreable — see the
    /// note in the body.
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

            // `layer.rank` is intentionally unbounded relative to
            // `min(d_in, d_out)`. The matrix rank of `B @ A` cannot exceed
            // that bound however wide the inner dimension is, so a
            // factorization above it is redundant: the same update is
            // exactly representable at `rank = min(d_in, d_out)`, and the
            // extra columns cost additional multiply-adds per hooked row in
            // `apply_lora`. That is an efficiency property of the caller's
            // factorization, not a validity condition — `apply_lora` derives
            // every buffer offset from `rank` itself, and the per-layer
            // buffer-length invariant enforced by `Self::new`
            // (`rank * d_in == a.len()`, `d_out * rank == b.len()`) is what
            // keeps those offsets in bounds. `blend_lora_adapters` produces
            // over-complete adapters by construction (the blended rank is
            // the sum of the source ranks), so rejecting them here would
            // make an adapter this crate can build unscoreable.
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
mod tests;
