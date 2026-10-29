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
