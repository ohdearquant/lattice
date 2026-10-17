//! Attention module index for `decode`, `differential`, `flash`, `flash_causal`, `gated`, `gdn`, `gdn_backward`, `gdn_fused`, `gqa`, `native_sparse`, and `standard`; re-exports the standard attention helpers.
pub mod decode;
pub mod differential;
pub mod flash;
pub mod flash_causal;
pub mod gated;
pub mod gdn;
#[cfg(feature = "train-backward")]
pub mod gdn_backward;
pub mod gdn_fused;
pub mod gqa;
pub mod native_sparse;
pub mod softmax_row;
pub mod standard;

// Re-export from standard for backward compat
pub use self::standard::*;

/// Stable tag enum for ADR-059 attention variants (consumed by ADR-060 pruning).
///
/// `AttentionTag` is the typed taxonomy that higher layers (`LayerStats`,
/// `CalibrationObserver`) use instead of the lossy string `name()`. Tag values
/// are stable; variant names here match ADR-059's `AttentionTag` identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttentionTag {
    Mha,
    Gqa,
    /// Tiled Flash Attention v2 (CPU, O(1) memory, no mask).
    FlashCpu,
    /// Tiled Flash Attention v2 with causal mask (decoder prefill).
    FlashCausal,
    Gdn,
    GdnFused,
    GatedGqa,
    Differential,
    /// Native Sparse Attention (ADR-059: sparse-hybrid state).
    Nsa,
    Decode,
}

/// Phase-1 metadata enum for ADR-059 attention dispatch.
///
/// Each variant names one attention mechanism supported by Lattice.
/// Variants that carry configuration embed a `Copy` config struct so callers
/// can cheaply copy the enum and inspect parameters without heap allocation.
///
/// This enum does **not** implement `AttentionOp`, allocate state/scratch, or
/// call kernels. Those belong to ADR-059 P2+. `AttentionKind` is the
/// *routing* and *metadata* layer — not the *execution* layer.
///
/// # ADR-059 Phase 1 landing
///
/// This type is deliberately wired ahead of the dispatch sites. Production
/// callers (`TransformerLayer`, Metal dispatch) will reference `AttentionKind`
/// in subsequent phases (P2+). The enum is intentionally dead from a call-site
/// perspective in P1; see ADR-059 §Implementation Phases.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AttentionKind {
    /// Standard multi-head attention (BERT-style bidirectional encoder).
    ///
    /// Uses pre-allocated `AttentionBuffers` scratch; no KV cache.
    Mha,
    /// Grouped Query Attention (Qwen3 / Llama-style causal decoder).
    ///
    /// Carries the head-count configuration needed for dispatch.
    Gqa(gqa::GqaConfig),
    /// Flash Attention v2: tiled O(1)-memory attention without causal mask.
    Flash,
    /// Flash Attention v2 with causal mask for decoder-only prefill.
    FlashCausal,
    /// Gated Delta Network: linear recurrent attention (Qwen3.5 linear layers).
    Gdn,
    /// SIMD-fused GDN: AVX2/NEON-accelerated alternative to `Gdn`.
    GdnFused,
    /// Gated GQA tag for Qwen3.5 full-attention layers; concrete GQA dimensions
    /// are supplied by the future `LayerSpec`/runtime wrapper.
    GatedGqa,
    /// Differential Attention: dual-softmax subtraction (Ye et al., ICLR 2025).
    Differential,
    /// Native Sparse Attention: compression + selection + sliding window (Yuan et al., ACL 2025).
    NativeSparse,
    /// Decode-optimized single-token fast-path tag; concrete GQA dimensions are
    /// supplied by the decode runtime.
    Decode,
}

impl AttentionKind {
    /// Human-readable name for logging and diagnostics.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            AttentionKind::Mha => "mha",
            AttentionKind::Gqa(_) => "gqa",
            AttentionKind::Flash => "flash",
            AttentionKind::FlashCausal => "flash_causal",
            AttentionKind::Gdn => "gdn",
            AttentionKind::GdnFused => "gdn_fused",
            AttentionKind::GatedGqa => "gated_gqa",
            AttentionKind::Differential => "differential",
            AttentionKind::NativeSparse => "native_sparse",
            AttentionKind::Decode => "decode",
        }
    }

    /// Returns the stable typed tag for this variant (ADR-059 taxonomy).
    ///
    /// Use this instead of `name()` when interfacing with ADR-060 pruning
    /// (`LayerStats`, `CalibrationObserver`), Metal kernel selection, or any
    /// code that needs a stable, matchable identifier.
    #[inline]
    pub fn tag(&self) -> AttentionTag {
        match self {
            AttentionKind::Mha => AttentionTag::Mha,
            AttentionKind::Gqa(_) => AttentionTag::Gqa,
            AttentionKind::Flash => AttentionTag::FlashCpu,
            AttentionKind::FlashCausal => AttentionTag::FlashCausal,
            AttentionKind::Gdn => AttentionTag::Gdn,
            AttentionKind::GdnFused => AttentionTag::GdnFused,
            AttentionKind::GatedGqa => AttentionTag::GatedGqa,
            AttentionKind::Differential => AttentionTag::Differential,
            AttentionKind::NativeSparse => AttentionTag::Nsa,
            AttentionKind::Decode => AttentionTag::Decode,
        }
    }

    /// Returns `true` if this attention variant applies a causal mask.
    ///
    /// Causal attention ensures position `i` cannot attend to position `j > i`.
    /// Bidirectional variants (MHA encoder, plain Flash) return `false`.
    #[inline]
    pub fn is_causal(&self) -> bool {
        match self {
            // Bidirectional encoder — no causal mask.
            AttentionKind::Mha => false,
            // Causal decoder attention.
            AttentionKind::Gqa(_) => true,
            // Flash without mask — full bidirectional.
            AttentionKind::Flash => false,
            // Flash with causal mask for decoder prefill.
            AttentionKind::FlashCausal => true,
            // GDN is a recurrent causal mechanism by construction.
            AttentionKind::Gdn => true,
            AttentionKind::GdnFused => true,
            // Gated GQA wraps GQA which is causal.
            AttentionKind::GatedGqa => true,
            // Differential attention uses causal softmax (Ye et al. §3).
            AttentionKind::Differential => true,
            // NSA three branches all enforce causal ordering (Yuan et al. §3).
            AttentionKind::NativeSparse => true,
            // Single-token decode always causal (attends only past KV).
            AttentionKind::Decode => true,
        }
    }

    /// Returns `true` if the **current** implementation reads from / writes to
    /// a KV cache.
    ///
    /// This reflects what the existing Rust kernels in this crate actually do
    /// today, not ADR-059 target-state intent. When P2 wires cache-backed
    /// wrappers for Flash/Differential/NSA, these values will be re-evaluated.
    ///
    /// - `Mha`: scratch buffers recomputed each pass — no KV cache.
    /// - `Gqa`/`GatedGqa`/`Decode`: append to and read from `FlatKVCache`.
    /// - `Gdn`/`GdnFused`: recurrent state matrix (`S`), not a KV cache.
    /// - `Flash`/`FlashCausal`: current tiled CPU kernels recompute K/V from
    ///   hidden states; no production cache path exists yet.
    /// - `Differential`/`NativeSparse`: standalone kernels consume
    ///   caller-supplied buffers; no production cache owner exists yet.
    #[inline]
    pub fn supports_kv_cache(&self) -> bool {
        match self {
            // MHA uses scratch buffers, not KV cache.
            AttentionKind::Mha => false,
            // Production Qwen generation appends K/V to FlatKVCache.
            AttentionKind::Gqa(_) => true,
            // Current tiled Flash path recomputes K/V from hidden states; P2 adds cache wrapper.
            AttentionKind::Flash => false,
            // Current causal Flash kernel is prefill over caller-supplied full K/V buffers.
            AttentionKind::FlashCausal => false,
            // GDN keeps a recurrent state (S matrix), not a growing KV cache.
            AttentionKind::Gdn => false,
            AttentionKind::GdnFused => false,
            // Qwen3.5 full-attention path uses KV cache (gated.rs wraps GQA + sigmoid gate).
            AttentionKind::GatedGqa => true,
            // Standalone differential kernel; no production cache path yet.
            AttentionKind::Differential => false,
            // NSA kernels consume projected branch buffers; no persistent cache state yet.
            AttentionKind::NativeSparse => false,
            // Decode path reads from an existing KV cache.
            AttentionKind::Decode => true,
        }
    }
}

#[cfg(test)]
mod attention_kind_tests;
