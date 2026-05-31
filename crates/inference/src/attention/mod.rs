pub mod decode;
pub mod differential;
pub mod flash;
pub mod flash_causal;
pub mod gated;
pub mod gdn;
pub mod gdn_fused;
pub mod gqa;
pub mod native_sparse;
pub mod standard;

// Re-export from standard for backward compat
pub use self::standard::*;

/// Stable tag enum for ADR-059 attention variants (consumed by ADR-060 pruning).
///
/// `AttentionTag` is the typed taxonomy that higher layers (`LayerStats`,
/// `CalibrationObserver`) use instead of the lossy string `name()`. Tag values
/// are stable; variant names here match ADR-059's `AttentionTag` identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
mod attention_kind_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // name()
    // -----------------------------------------------------------------------

    #[test]
    fn name_mha() {
        assert_eq!(AttentionKind::Mha.name(), "mha");
    }

    #[test]
    fn name_gqa() {
        let cfg = gqa::GqaConfig {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        };
        assert_eq!(AttentionKind::Gqa(cfg).name(), "gqa");
    }

    #[test]
    fn name_flash() {
        assert_eq!(AttentionKind::Flash.name(), "flash");
    }

    #[test]
    fn name_flash_causal() {
        assert_eq!(AttentionKind::FlashCausal.name(), "flash_causal");
    }

    #[test]
    fn name_gdn() {
        assert_eq!(AttentionKind::Gdn.name(), "gdn");
    }

    #[test]
    fn name_gdn_fused() {
        assert_eq!(AttentionKind::GdnFused.name(), "gdn_fused");
    }

    #[test]
    fn name_gated_gqa() {
        assert_eq!(AttentionKind::GatedGqa.name(), "gated_gqa");
    }

    #[test]
    fn name_differential() {
        assert_eq!(AttentionKind::Differential.name(), "differential");
    }

    #[test]
    fn name_native_sparse() {
        assert_eq!(AttentionKind::NativeSparse.name(), "native_sparse");
    }

    #[test]
    fn name_decode() {
        assert_eq!(AttentionKind::Decode.name(), "decode");
    }

    // -----------------------------------------------------------------------
    // tag()
    // -----------------------------------------------------------------------

    #[test]
    fn tag_mha() {
        assert_eq!(AttentionKind::Mha.tag(), AttentionTag::Mha);
    }

    #[test]
    fn tag_gqa() {
        let cfg = gqa::GqaConfig {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        };
        assert_eq!(AttentionKind::Gqa(cfg).tag(), AttentionTag::Gqa);
    }

    #[test]
    fn tag_flash_maps_to_flash_cpu() {
        assert_eq!(AttentionKind::Flash.tag(), AttentionTag::FlashCpu);
    }

    #[test]
    fn tag_flash_causal() {
        assert_eq!(AttentionKind::FlashCausal.tag(), AttentionTag::FlashCausal);
    }

    #[test]
    fn tag_gdn() {
        assert_eq!(AttentionKind::Gdn.tag(), AttentionTag::Gdn);
    }

    #[test]
    fn tag_gdn_fused() {
        assert_eq!(AttentionKind::GdnFused.tag(), AttentionTag::GdnFused);
    }

    #[test]
    fn tag_gated_gqa() {
        assert_eq!(AttentionKind::GatedGqa.tag(), AttentionTag::GatedGqa);
    }

    #[test]
    fn tag_differential() {
        assert_eq!(
            AttentionKind::Differential.tag(),
            AttentionTag::Differential
        );
    }

    #[test]
    fn tag_native_sparse_maps_to_nsa() {
        assert_eq!(AttentionKind::NativeSparse.tag(), AttentionTag::Nsa);
    }

    #[test]
    fn tag_decode() {
        assert_eq!(AttentionKind::Decode.tag(), AttentionTag::Decode);
    }

    // -----------------------------------------------------------------------
    // is_causal()
    // -----------------------------------------------------------------------

    #[test]
    fn is_causal_mha_false() {
        assert!(!AttentionKind::Mha.is_causal());
    }

    #[test]
    fn is_causal_gqa_true() {
        let cfg = gqa::GqaConfig {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        };
        assert!(AttentionKind::Gqa(cfg).is_causal());
    }

    #[test]
    fn is_causal_flash_false() {
        assert!(!AttentionKind::Flash.is_causal());
    }

    #[test]
    fn is_causal_flash_causal_true() {
        assert!(AttentionKind::FlashCausal.is_causal());
    }

    #[test]
    fn is_causal_gdn_true() {
        assert!(AttentionKind::Gdn.is_causal());
    }

    #[test]
    fn is_causal_gdn_fused_true() {
        assert!(AttentionKind::GdnFused.is_causal());
    }

    #[test]
    fn is_causal_gated_gqa_true() {
        assert!(AttentionKind::GatedGqa.is_causal());
    }

    #[test]
    fn is_causal_differential_true() {
        assert!(AttentionKind::Differential.is_causal());
    }

    #[test]
    fn is_causal_native_sparse_true() {
        assert!(AttentionKind::NativeSparse.is_causal());
    }

    #[test]
    fn is_causal_decode_true() {
        assert!(AttentionKind::Decode.is_causal());
    }

    // -----------------------------------------------------------------------
    // supports_kv_cache() — reflects current implementation, not ADR target state
    // -----------------------------------------------------------------------

    #[test]
    fn kv_cache_mha_false() {
        assert!(!AttentionKind::Mha.supports_kv_cache());
    }

    #[test]
    fn kv_cache_gqa_true() {
        let cfg = gqa::GqaConfig {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        };
        assert!(AttentionKind::Gqa(cfg).supports_kv_cache());
    }

    #[test]
    fn kv_cache_flash_false() {
        // Current tiled Flash CPU kernel recomputes K/V from hidden states; no cache path.
        assert!(!AttentionKind::Flash.supports_kv_cache());
    }

    #[test]
    fn kv_cache_flash_causal_false() {
        // Prefill over caller-supplied full K/V buffers; no production cache owner.
        assert!(!AttentionKind::FlashCausal.supports_kv_cache());
    }

    #[test]
    fn kv_cache_gdn_false() {
        // GDN uses a recurrent state matrix, not a growing KV cache.
        assert!(!AttentionKind::Gdn.supports_kv_cache());
    }

    #[test]
    fn kv_cache_gdn_fused_false() {
        assert!(!AttentionKind::GdnFused.supports_kv_cache());
    }

    #[test]
    fn kv_cache_gated_gqa_true() {
        assert!(AttentionKind::GatedGqa.supports_kv_cache());
    }

    #[test]
    fn kv_cache_differential_false() {
        // Standalone kernel; no production cache state.
        assert!(!AttentionKind::Differential.supports_kv_cache());
    }

    #[test]
    fn kv_cache_native_sparse_false() {
        // NSA kernels consume projected branch buffers; no persistent cache yet.
        assert!(!AttentionKind::NativeSparse.supports_kv_cache());
    }

    #[test]
    fn kv_cache_decode_true() {
        assert!(AttentionKind::Decode.supports_kv_cache());
    }

    // -----------------------------------------------------------------------
    // Clone/Copy round-trip (Gqa carries a Copy config — verify no drop issues)
    // -----------------------------------------------------------------------

    #[test]
    fn clone_gqa_preserves_config() {
        let cfg = gqa::GqaConfig {
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let kind = AttentionKind::Gqa(cfg);
        let cloned = kind;
        assert_eq!(cloned.name(), "gqa");
        if let AttentionKind::Gqa(c) = cloned {
            assert_eq!(c.num_heads, 32);
            assert_eq!(c.num_kv_heads, 4);
            assert_eq!(c.head_dim, 128);
        } else {
            panic!("clone changed variant");
        }
    }

    // -----------------------------------------------------------------------
    // Exhaustiveness: all 10 variants covered by name(), tag(), is_causal(),
    // supports_kv_cache(). Compile-time check via a match with no wildcard.
    // -----------------------------------------------------------------------

    #[test]
    fn all_variants_have_names() {
        let cfg = gqa::GqaConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
        };
        let variants: &[AttentionKind] = &[
            AttentionKind::Mha,
            AttentionKind::Gqa(cfg),
            AttentionKind::Flash,
            AttentionKind::FlashCausal,
            AttentionKind::Gdn,
            AttentionKind::GdnFused,
            AttentionKind::GatedGqa,
            AttentionKind::Differential,
            AttentionKind::NativeSparse,
            AttentionKind::Decode,
        ];
        assert_eq!(variants.len(), 10, "update test when adding a new variant");
        for v in variants {
            assert!(!v.name().is_empty());
            // tag() must compile (exhaustive match — adding a variant without
            // updating tag() is a compile error, not a runtime error).
            let _ = v.tag();
        }
    }
}
