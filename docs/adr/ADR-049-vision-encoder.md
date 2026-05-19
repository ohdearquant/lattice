# ADR-049: Vision Encoder Integration (Qwen-VL Path)

**Status**: Proposed
**Date**: 2026-05-19
**Crate**: lattice-inference

## Context

Lattice currently processes only text: `BpeTokenizer` converts strings to token IDs, and
`MetalQwen35State` runs a text-only decoder. There is no image ingestion path, no vision encoder,
and no mechanism to prepend patch tokens to the text sequence.

The natural extension is the Qwen-VL family. Lattice already targets Qwen3.5 for text; the
corresponding multimodal variants are Qwen2.5-VL (arxiv:2502.13923) and Qwen3-VL.

**Qwen2.5-VL** (arxiv:2502.13923): ViT with window attention + 2D RoPE, MLP merger, decoder LLM.
Depth 32, patch size 14, temporal_patch_size 2.

**Qwen3-VL** (Hugging Face `Qwen3VLVisionConfig` defaults): ViT depth 27, patch size 16,
spatial_merge_size 2, MLP-based merger (no cross-attention adapter). Shares the Qwen3.5 decoder
backbone.

Both variants output a flat sequence of patch embedding vectors that are prepended to the text
token embeddings in the decoder's residual stream. The MLP merger is approximately 10 M parameters
— negligible relative to the decoder.

This ADR targets **Qwen3-VL** as v0, using its actual Hugging Face config (depth 27, patch 16),
not the Qwen2.5-VL architecture.

The competitive baseline is mature: llama.cpp ships `libmtmd` supporting Qwen2.5-VL, Qwen3 Omni,
Gemma3/4, Mistral Small 3.1, and Pixtral. Ollama exposes vision via its existing API. MLX has
`mlx-vlm`. Lattice needs a clear integration path to remain relevant for multimodal workloads on
Apple Silicon.

Metal GPU can execute both the ViT encoder and the decoder without device transfer. The existing
`MetalQwen35State` handles the LLM side; the only additions are image preprocessing, ViT forward
pass, MLP projection, and sequence splicing.

## Decision

For v0, implement the Qwen-VL integration path inside `lattice-inference` as a new
`vision` module. The scope is narrow and additive: the decoder is not modified. The vision module
provides image preprocessing, a ViT forward pass over Metal, a two-layer MLP projector, and a
`MultimodalInput` type that carries pre-projected patch embeddings alongside text tokens.

The ViT weights are loaded from the Qwen3-VL safetensors checkpoint (same loading path as the
decoder via ADR-003). No external image processing library is linked at the Metal kernel level;
standard Rust `image` crate handles CPU-side resize and normalize before patch extraction.

The integration point is `MetalQwen35State`: a new `generate_multimodal` entry point accepts
`MultimodalInput`, prepends patch embeddings to the token embedding matrix, and delegates to the
existing decode loop. The existing `generate_greedy` / `generate_sampled` paths are not changed.

**Fixed resolution for v0**: 448 × 448 pixels, 14 × 14 patch size, 1024 patches per image.
Dynamic resolution (Qwen2.5-VL's native tiling scheme) is deferred to v1.

**Single VLM family for v0**: Qwen3-VL only. Generalizing to SigLIP, InternViT, or other ViT
backbones is deferred until the integration pattern is validated.

## Scope

**v0 (this ADR)**:
- `vision/preprocess.rs`: CPU-side image resize (bilinear), per-channel normalize, patch grid
  extraction. Output: `Vec<f32>` shaped `[n_patches, patch_h, patch_w, 3]`.
- `vision/vit.rs`: ViT encoder forward pass on Metal. Implements the Qwen3-VL ViT configuration
  (patch size 14, window attention, 32 transformer blocks at the 7 B scale).
- `vision/mlp_merger.rs`: Two-layer MLP projector on CPU (small enough that CPU is adequate).
  Input: `[n_patches, d_vit]`. Output: `[n_patches, d_model]` where `d_model` matches the
  decoder hidden dimension.
- `vision/mod.rs`: `MultimodalInput { text_tokens: Vec<u32>, patch_embeddings: Array2<f32> }`
  and `VisionEncoder::encode(image_bytes) -> Result<Array2<f32>>`.
- `metal_qwen35.rs`: `generate_multimodal(input: MultimodalInput, config: GenerateConfig)`
  entry point. Splices patch embeddings before position 0 of the token embedding sequence.
- Weight loading: extend `load_qwen35_weights` to detect and load `vision_model.*` keys from
  the VLM safetensors file. No new loader required.

**Deferred**:
- Dynamic resolution tiling (Qwen2.5-VL style — arbitrary input size tiled to NxM crops)
- Multi-image inputs (more than one image per prompt)
- Video frame input
- SigLIP or InternViT as alternative ViT backends
- Image caching across decode steps (KV-cache for the ViT output)
- CPU fallback path for the ViT (v0 is Metal-only as an experimental feature; a CPU ViT
  forward pass is straightforward to add as a follow-up since all ops are standard linear algebra)
- Vision fine-tuning or LoRA injection into the ViT

## Architecture

### Core Types

```rust
// vision/mod.rs

/// Preprocessed image ready for the ViT encoder.
pub struct ImageTensor {
    /// Row-major float32 pixel data: [n_patches, patch_h * patch_w * 3].
    pub patches: Vec<f32>,
    pub n_patches: usize,
    pub patch_hw: usize,  // patch_h == patch_w == 14
}

/// A multimodal prompt: patch embeddings prepended to text tokens.
pub struct MultimodalInput {
    /// Projected patch vectors: [n_patches, d_model].
    pub patch_embeddings: Vec<f32>,
    pub n_patches: usize,
    pub d_model: usize,
    /// Text token IDs following the image.
    pub text_tokens: Vec<u32>,
}

/// End-to-end vision encoder: image bytes → projected patch embeddings.
pub struct VisionEncoder {
    vit: ViT,
    mlp: MlpMerger,
}

impl VisionEncoder {
    pub fn new(weights: &VisionWeights, config: &ViTConfig) -> Result<Self, VisionError>;
    /// Encode raw image bytes (JPEG or PNG). Returns [n_patches, d_model].
    pub fn encode(&self, image_bytes: &[u8]) -> Result<Vec<f32>, VisionError>;
}
```

```rust
// vision/preprocess.rs

pub struct PreprocessConfig {
    pub image_size: u32,   // 448 for v0
    pub patch_size: u32,   // 14 for v0
    pub mean: [f32; 3],    // Qwen3-VL normalization constants
    pub std:  [f32; 3],
}

/// Decode + resize to `image_size × image_size`, normalize, extract patch grid.
/// Returns Vec<f32> shaped [n_patches, patch_size * patch_size * 3] (row-major).
pub fn preprocess(image_bytes: &[u8], cfg: &PreprocessConfig)
    -> Result<ImageTensor, VisionError>;
```

```rust
// vision/vit.rs  — Metal-backed ViT encoder

pub struct ViT {
    config: ViTConfig,
    // Metal buffers for patch embedding projection, attention, MLP weights.
    // Allocated once at construction; reused across encode calls.
    patch_embed: MetalLinear,
    blocks: Vec<ViTBlock>,
    norm: MetalLayerNorm,
}

pub struct ViTConfig {
    pub image_size: u32,         // 448
    pub patch_size: u32,         // 16 (Qwen3-VL default)
    pub n_patches: usize,        // (448/16)^2 = 784
    pub d_model: usize,          // 1152 for Qwen3-VL 7B
    pub n_heads: usize,          // 16
    pub n_layers: usize,         // 27 (Qwen3-VL default)
    pub spatial_merge_size: usize, // 2 (Qwen3-VL default)
    pub global_attn_every: usize,// every 4 blocks uses global attention
}

impl ViT {
    /// Forward pass. Input: `ImageTensor` (CPU). Output: `[n_patches, d_model]` (CPU).
    /// Internally allocates Metal command buffer, runs encoder, reads back to CPU.
    pub fn forward(&self, img: &ImageTensor) -> Result<Vec<f32>, VisionError>;
}
```

```rust
// vision/mlp_merger.rs  — CPU-side MLP projector

pub struct MlpMerger {
    w1: Vec<f32>,  // [d_vit, d_hidden]
    b1: Vec<f32>,
    w2: Vec<f32>,  // [d_hidden, d_model]
    b2: Vec<f32>,
    d_vit: usize,
    d_hidden: usize,
    d_model: usize,
}

impl MlpMerger {
    /// Project ViT output to decoder hidden dimension.
    /// Input: flat f32 slice [n_patches * d_vit].
    /// Output: Vec<f32> [n_patches * d_model].
    pub fn project(&self, vit_out: &[f32], n_patches: usize) -> Vec<f32>;
}
```

### Integration with MetalQwen35State

`MetalQwen35State` gains a new entry point:

```rust
impl MetalQwen35State {
    /// Multimodal generation. Patch embeddings are prepended to the token
    /// embedding sequence before the first decode step. Position IDs for
    /// the text tokens are offset by `input.n_patches`. All other decode
    /// parameters (KV cache, sampling, EOS) behave identically to
    /// `generate_greedy` / `generate_sampled`.
    pub fn generate_multimodal(
        &mut self,
        input: MultimodalInput,
        config: GenerateConfig,
    ) -> Result<Vec<u32>, InferenceError>;
}
```

Internally, `generate_multimodal` calls `embed_tokens` for the text portion and concatenates
the patch embeddings at position 0 before the first prefill call. RoPE position IDs start
at `n_patches` for the first text token, so image patches occupy positions `[0, n_patches)`.
This matches the Qwen3-VL positional encoding convention.

### Weight Loading

The Qwen3-VL checkpoint stores vision weights under `vision_model.*` keys in the safetensors
file. The existing `load_qwen35_weights` function (ADR-003) is extended with a
`load_vision_weights` pass that reads these keys into `VisionWeights`, which is then passed to
`VisionEncoder::new`. Models without `vision_model.*` keys (text-only Qwen3.5) load unchanged.

### Memory Budget (7 B model)

| Component | Parameters | fp16 size |
|---|---|---|
| ViT encoder | ~300 M | ~600 MB |
| MLP merger | ~10 M | ~20 MB |
| Decoder (Qwen3.5-7B) | ~7 B | ~14 GB |
| Patch KV activations (1024 patches) | — | ~50 MB |
| **Total** | | **~15 GB** |

Apple M2/M3/M4 Max with 48 GB unified memory accommodates this. The 16 GB base tier cannot;
v0 targets 32 GB+ configurations. A future 3 B VLM variant would target 16 GB.

## Alternatives Considered

**1. FFI to llama.cpp libmtmd**
llama.cpp's `libmtmd` is mature and supports 10+ VLM families. FFI is technically feasible.
Rejected: (a) violates the pure-Rust constraint (ADR-001); (b) introduces a C++ build dependency
that breaks the single-`cargo build` workflow; (c) llama.cpp's Metal path and Lattice's Metal
path would compete for the same GPU command queue without a coordination layer.

**2. MLX-VLM via subprocess**
`mlx-vlm` is a Python library with native Apple Silicon support. Spawning a subprocess for
inference contradicts Lattice's no-subprocess design and adds ~200 ms cold-start latency per
encode call. Rejected for architectural reasons.

**3. Load SigLIP weights instead of InternViT/Qwen ViT**
SigLIP (from Google) is used by LLaVA-style models and PaliGemma. It would require a separate
weight file and a different normalization convention. Since Lattice targets Qwen3-VL, using the
ViT already bundled in the Qwen3-VL checkpoint is the path of least resistance. Deferred as a
future alternative backend when multi-family support is warranted.

**4. Cross-attention adapter (Q-Former style)**
Qwen2-VL and BLIP-2 use a Q-Former or cross-attention adapter between the ViT and the decoder.
Qwen3-VL replaced this with the simpler MLP merger. The MLP path is implemented here; the
cross-attention design is explicitly not pursued for v0 since Qwen3-VL is the target.

**5. Dynamic resolution from v0**
Qwen2.5-VL tiles arbitrary-resolution images to NxM crops, enabling higher-resolution inputs.
This complicates sequence length management and KV cache sizing. Fixed 448×448 is sufficient for
standard benchmarks (TextVQA, DocVQA single-page) and lets v0 ship without dynamic shape logic.
Deferred to v1.

## Risks

**R1: ViT Metal kernel correctness**
The ViT uses window attention with periodic global attention layers — a non-standard attention
pattern. Mitigation: validate ViT output numerically against the Hugging Face PyTorch reference
(Qwen/Qwen3-VL-7B-Instruct) on a fixed test image; require cosine similarity > 0.999 between
Lattice and reference patch embeddings before merging.

**R2: Unified memory pressure at 32 GB**
The ViT (~600 MB) + decoder (~14 GB) + activations push close to 16 GB peak during prefill on
32 GB machines if Metal command buffers are allocated concurrently. Mitigation: encode the image
first, read ViT output back to CPU, release Metal ViT buffers, then run the decoder. Sequential
rather than concurrent GPU use keeps peak below 15 GB.

**R3: Positional encoding alignment**
Patch tokens use 2D RoPE in Qwen3-VL (row and column indices, not sequential position IDs).
The existing 1D RoPE implementation (ADR-007) cannot be reused for the visual tokens without
extension. Mitigation: patch tokens bypass RoPE in the decoder (position embeddings for image
tokens are applied inside the ViT, not re-applied in the decoder). Confirm this matches the
Qwen3-VL reference implementation before finalizing.

**R4: Weight key schema drift between Qwen2.5-VL and Qwen3-VL**
Safetensors key names differ between model versions. Mitigation: enumerate the actual key
prefixes for both model variants in the weight loader; emit a clear error if unrecognized
vision keys are encountered.

**R5: `image` crate decode performance**
The `image` crate (CPU decode + resize) adds ~5–20 ms per image. Acceptable for interactive
workloads. Not acceptable for video frame throughput. Mitigation: document the limitation;
defer Metal-accelerated image decode to v1 when video inputs are scoped.

## References

- Qwen2.5-VL paper: arxiv:2502.13923 — architecture overview, MLP merger design, window attention
- Qwen3-VL model card: huggingface.co/Qwen/Qwen3-VL-7B-Instruct
- llama.cpp libmtmd: `tools/mtmd/` in the llama.cpp repository — reference for VLM token splicing
- MLX-VLM: github.com/ml-explore/mlx-vlm — Apple Silicon reference implementation (Python)
- Prior lattice ADRs: ADR-001 (pure Rust), ADR-003 (safetensors loading), ADR-007 (RoPE),
  ADR-009 (model architectures), ADR-010 (attention mechanisms)
