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

/// Unified attention dispatch enum (ADR-059).
///
/// Each variant names one attention mechanism supported by Lattice.
/// Variants that carry configuration embed a `Copy` config struct so callers
/// can cheaply clone the enum and inspect parameters without heap allocation.
///
/// This enum does **not** own mutable state (scratch buffers, KV caches); those
/// remain in the per-variant structs in the individual modules. `AttentionKind`
/// is the *routing* layer, not the *execution* layer.
#[derive(Debug, Clone)]
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
    /// Gated GQA: per-element sigmoid gate applied after GQA context aggregation.
    GatedGqa,
    /// Differential Attention: dual-softmax subtraction (Ye et al., ICLR 2025).
    Differential,
    /// Native Sparse Attention: compression + selection + sliding window (Yuan et al., ACL 2025).
    NativeSparse,
    /// Decode-optimized single-token fast path using `GqaConfig`.
    Decode,
}

impl AttentionKind {
    /// Human-readable name for logging and diagnostics.
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

    /// Returns `true` if this attention variant applies a causal mask.
    ///
    /// Causal attention ensures position `i` cannot attend to position `j > i`.
    /// Bidirectional variants (MHA encoder, plain Flash) return `false`.
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

    /// Returns `true` if this variant reads from / writes to a KV cache.
    ///
    /// MHA uses pre-allocated scratch buffers recomputed each forward pass and
    /// does not maintain a KV cache. KV-backed decoder variants maintain
    /// past-KV state; GDN variants use recurrent state matrices instead.
    pub fn supports_kv_cache(&self) -> bool {
        match self {
            // MHA uses scratch buffers, not KV cache.
            AttentionKind::Mha => false,
            AttentionKind::Gqa(_) => true,
            // Flash CPU uses a KV cache (ADR-059 state table: Flash CPU → Softmax, KV-backed).
            AttentionKind::Flash => true,
            AttentionKind::FlashCausal => true,
            // GDN keeps a recurrent state (S matrix), not a growing KV cache.
            AttentionKind::Gdn => false,
            AttentionKind::GdnFused => false,
            AttentionKind::GatedGqa => true,
            AttentionKind::Differential => true,
            AttentionKind::NativeSparse => true,
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
    // supports_kv_cache()
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
    fn kv_cache_flash_true() {
        // ADR-059 state table classifies Flash CPU as KV-backed (Softmax category).
        assert!(AttentionKind::Flash.supports_kv_cache());
    }

    #[test]
    fn kv_cache_flash_causal_true() {
        assert!(AttentionKind::FlashCausal.supports_kv_cache());
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
    fn kv_cache_differential_true() {
        assert!(AttentionKind::Differential.supports_kv_cache());
    }

    #[test]
    fn kv_cache_native_sparse_true() {
        assert!(AttentionKind::NativeSparse.supports_kv_cache());
    }

    #[test]
    fn kv_cache_decode_true() {
        assert!(AttentionKind::Decode.supports_kv_cache());
    }

    // -----------------------------------------------------------------------
    // Clone round-trip (Gqa carries a Copy config — verify no drop issues)
    // -----------------------------------------------------------------------

    #[test]
    fn clone_gqa_preserves_config() {
        let cfg = gqa::GqaConfig {
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let kind = AttentionKind::Gqa(cfg);
        let cloned = kind.clone();
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
    // Exhaustiveness: all 10 variants covered by name(), is_causal(),
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
        }
    }
}
