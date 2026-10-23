//! Size bounds, byte budgets and value checks enforced when a config is validated.

use super::*;

// ──────────────────────────────────────────────────────────────────────
// GatedDeltaNet state-matrix three-factor budget (MAX_GDN_STATE_SIZE)
// ──────────────────────────────────────────────────────────────────────

/// `MAX_LINEAR_OUTPUT_DIM` only bounds `value_heads * linear_value_head_dim`;
/// `linear_key_head_dim` is a free multiplier on top of it in `attention/gdn.rs`'s
/// `s_matrices` allocation. `value_heads = 1`, `linear_value_head_dim` just under the
/// `MAX_LINEAR_OUTPUT_DIM` cap, and a huge `linear_key_head_dim` must pass
/// `MAX_LINEAR_OUTPUT_DIM` (linear_output_dim = value_dim, well under the cap) while still
/// being rejected by the three-factor `MAX_GDN_STATE_SIZE` budget.
#[test]
fn parser_rejects_gdn_state_size_over_max_with_small_output_dim() {
    let json = r#"{"text_config": {
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 1000000,
            "linear_value_head_dim": 128
        }}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err(
            "a huge linear_key_head_dim must be rejected by MAX_GDN_STATE_SIZE even though \
                 linear_output_dim (value_heads * linear_value_head_dim) is tiny",
        )
        .to_string();
    assert!(
        err.contains("GatedDeltaNet state size") && err.contains("MAX_GDN_STATE_SIZE"),
        "wrong guard fired: {err}"
    );
}

/// Accept-case control: realistic GDN geometry (matching the preset) must still parse.
#[test]
fn parser_accepts_realistic_gdn_state_size() {
    let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128
        }}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "realistic GDN geometry must be accepted"
    );
}

/// Zero `linear_key_head_dim` is a degenerate zero-width GDN recurrence.
#[test]
fn parser_rejects_zero_linear_key_head_dim() {
    let json = r#"{"text_config": {"linear_key_head_dim": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("linear_key_head_dim: 0 must be rejected")
        .to_string();
    assert!(
        err.contains("linear_key_head_dim"),
        "wrong guard fired: {err}"
    );
}

/// Zero `linear_value_head_dim` is a degenerate zero-width GDN recurrence, and would
/// otherwise collapse `linear_output_dim()` to zero -- passing `MAX_LINEAR_OUTPUT_DIM`
/// trivially rather than being caught by it.
#[test]
fn parser_rejects_zero_linear_value_head_dim() {
    let json = r#"{"text_config": {"linear_value_head_dim": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("linear_value_head_dim: 0 must be rejected")
        .to_string();
    assert!(
        err.contains("linear_value_head_dim"),
        "wrong guard fired: {err}"
    );
}

// ──────────────────────────────────────────────────────────────────────
// hidden_size upper bound (MAX_HIDDEN_SIZE)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn parser_rejects_hidden_size_over_max() {
    let json = format!(
        r#"{{"text_config": {{"hidden_size": {}}}}}"#,
        MAX_HIDDEN_SIZE + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("hidden_size above MAX_HIDDEN_SIZE must be rejected")
        .to_string();
    assert!(
        err.contains("hidden_size") && err.contains("MAX_HIDDEN_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_hidden_size_at_max() {
    // hidden_size alone (with vocab_size at its own realistic-preset value) does not
    // reach MAX_EMBEDDING_BYTES -- unlike the vision "_at_max" cases below, the text
    // embedding budget is generous enough (32 GiB) to keep this at-max-in-isolation case
    // accepted. See MAX_EMBEDDING_BYTES docs. intermediate_size is pinned to 1 (FIX 8:
    // hidden_size alone at MAX_HIDDEN_SIZE, paired with the preset's own default
    // intermediate_size, would otherwise breach the new MAX_DENSE_FFN_TENSOR_BYTES
    // product budget -- this test is about the per-field hidden_size cap in isolation).
    let json = format!(
        r#"{{"text_config": {{"hidden_size": {MAX_HIDDEN_SIZE}, "vocab_size": 1, "intermediate_size": 1}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "hidden_size == MAX_HIDDEN_SIZE must be accepted when paired with a small vocab_size"
    );
}

// ──────────────────────────────────────────────────────────────────────
// attention head-count upper bound (MAX_ATTENTION_HEADS)
// ──────────────────────────────────────────────────────────────────────

/// `num_attention_heads` drives `scores = num_attention_heads * (max_kv_len + 1)`
/// independent of `head_dim`; a small `head_dim` (1) paired with a huge
/// `num_attention_heads` still passes `MAX_FULL_ATTENTION_DIM` (the product stays small)
/// so `MAX_ATTENTION_HEADS` must fire on its own.
#[test]
fn parser_rejects_num_attention_heads_over_max() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": 1, "num_attention_heads": {}, "num_key_value_heads": 1}}}}"#,
        MAX_ATTENTION_HEADS + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("num_attention_heads above MAX_ATTENTION_HEADS must be rejected")
        .to_string();
    assert!(
        err.contains("num_attention_heads") && err.contains("MAX_ATTENTION_HEADS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_num_attention_heads_at_max() {
    // head_dim=8 with the default partial_rotary_factor (0.25) gives rope_dim=2, the
    // smallest value that clears the `rope_dim >= 2` guard -- head_dim=1 would derive
    // rope_dim=0 and fail that unrelated check instead of exercising this bound.
    let json = format!(
        r#"{{"text_config": {{"head_dim": 8, "num_attention_heads": {MAX_ATTENTION_HEADS}, "num_key_value_heads": 1}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "num_attention_heads == MAX_ATTENTION_HEADS must be accepted"
    );
}

/// `num_attention_heads` must be divisible by `num_key_value_heads`, which forces
/// `num_attention_heads >= num_key_value_heads` -- so isolating this bound requires
/// `num_attention_heads` to stay within its own budget while `num_key_value_heads` alone
/// exceeds it (the reverse combination is unreachable: an out-of-budget
/// `num_key_value_heads` would always drag `num_attention_heads` out of budget too).
#[test]
fn parser_rejects_num_key_value_heads_over_max() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": 1, "num_attention_heads": {MAX_ATTENTION_HEADS}, "num_key_value_heads": {}}}}}"#,
        MAX_ATTENTION_HEADS + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("num_key_value_heads above MAX_ATTENTION_HEADS must be rejected")
        .to_string();
    assert!(
        err.contains("num_key_value_heads") && err.contains("MAX_ATTENTION_HEADS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_num_key_value_heads_at_max() {
    // head_dim=8: see parser_accepts_num_attention_heads_at_max for why not 1.
    let json = format!(
        r#"{{"text_config": {{"head_dim": 8, "num_attention_heads": {MAX_ATTENTION_HEADS}, "num_key_value_heads": {MAX_ATTENTION_HEADS}}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "num_key_value_heads == MAX_ATTENTION_HEADS must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// num_experts_per_tok upper bound independent of num_experts presence
// ──────────────────────────────────────────────────────────────────────

/// `router_selected.resize(cfg.num_experts_per_tok.unwrap_or(0), ..)` runs regardless of
/// whether `num_experts` is present; a checkpoint that sets `num_experts_per_tok` while
/// leaving `num_experts` unset must still be bounded.
#[test]
fn parser_rejects_num_experts_per_tok_over_max_without_num_experts() {
    let json = format!(
        r#"{{"text_config": {{"num_experts": null, "num_experts_per_tok": {}}}}}"#,
        MAX_NUM_EXPERTS + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "num_experts_per_tok above MAX_NUM_EXPERTS must be rejected even when \
                 num_experts is absent",
        )
        .to_string();
    assert!(
        err.contains("num_experts_per_tok") && err.contains("MAX_NUM_EXPERTS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_num_experts_per_tok_at_max_without_num_experts() {
    let json = format!(
        r#"{{"text_config": {{"num_experts": null, "num_experts_per_tok": {MAX_NUM_EXPERTS}}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "num_experts_per_tok == MAX_NUM_EXPERTS must be accepted when num_experts is absent"
    );
}

// ──────────────────────────────────────────────────────────────────────
// nonzero lower bounds: intermediate_size / moe / shared-expert intermediate sizes
// ──────────────────────────────────────────────────────────────────────

#[test]
fn parser_rejects_zero_intermediate_size() {
    let json = r#"{"text_config": {"intermediate_size": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("intermediate_size: 0 must be rejected")
        .to_string();
    assert!(
        err.contains("intermediate_size"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_zero_moe_intermediate_size() {
    let json = r#"{"text_config": {"moe_intermediate_size": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("moe_intermediate_size: 0 must be rejected")
        .to_string();
    assert!(
        err.contains("moe_intermediate_size"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_zero_shared_expert_intermediate_size() {
    let json = r#"{"text_config": {"shared_expert_intermediate_size": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("shared_expert_intermediate_size: 0 must be rejected")
        .to_string();
    assert!(
        err.contains("shared_expert_intermediate_size"),
        "wrong guard fired: {err}"
    );
}

// ──────────────────────────────────────────────────────────────────────
// max_position_embeddings upper bound (MAX_POSITION_EMBEDDINGS)
// ──────────────────────────────────────────────────────────────────────

/// Drives the Metal RoPE table allocation (`rope_max * rope_dim / 2`), independent of
/// `head_dim`.
#[test]
fn parser_rejects_max_position_embeddings_over_max() {
    // `head_dim`/`partial_rotary_factor` pinned to the smallest valid `rope_dim` (2) so
    // this trips only the `max_position_embeddings`-specific guard, not the
    // `MAX_ROPE_TABLE_BYTES` product budget (which the default preset's `rope_dim = 64`
    // would otherwise reach first at this `max_position_embeddings`).
    let json = format!(
        r#"{{"text_config": {{"max_position_embeddings": {}, "head_dim": 2, "partial_rotary_factor": 1.0}}}}"#,
        MAX_POSITION_EMBEDDINGS + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("max_position_embeddings above MAX_POSITION_EMBEDDINGS must be rejected")
        .to_string();
    assert!(
        err.contains("max_position_embeddings") && err.contains("MAX_POSITION_EMBEDDINGS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_max_position_embeddings_at_max() {
    let json =
        format!(r#"{{"text_config": {{"max_position_embeddings": {MAX_POSITION_EMBEDDINGS}}}}}"#);
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "max_position_embeddings == MAX_POSITION_EMBEDDINGS must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// linear_num_key_heads upper bound (MAX_LINEAR_NUM_KEY_HEADS)
// ──────────────────────────────────────────────────────────────────────

/// `linear_num_key_heads` was previously checked only for zero; it is a free multiplier
/// in `linear_qkv_dim()` unconstrained by `MAX_LINEAR_OUTPUT_DIM` / `MAX_GDN_STATE_SIZE`
/// (neither product carries it as a factor).
#[test]
fn parser_rejects_linear_num_key_heads_over_max() {
    let json = format!(
        r#"{{"text_config": {{
                "linear_num_key_heads": {over},
                "linear_num_value_heads": {over},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#,
        over = MAX_LINEAR_NUM_KEY_HEADS + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("linear_num_key_heads above MAX_LINEAR_NUM_KEY_HEADS must be rejected")
        .to_string();
    assert!(
        err.contains("linear_num_key_heads") && err.contains("MAX_LINEAR_NUM_KEY_HEADS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_linear_num_key_heads_at_max() {
    let json = format!(
        r#"{{"text_config": {{
                "linear_num_key_heads": {MAX_LINEAR_NUM_KEY_HEADS},
                "linear_num_value_heads": {MAX_LINEAR_NUM_KEY_HEADS},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "linear_num_key_heads == MAX_LINEAR_NUM_KEY_HEADS must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// linear_conv_kernel_dim upper bound (MAX_CONV_KERNEL_DIM)
// ──────────────────────────────────────────────────────────────────────

/// `linear_conv_kernel_dim` was previously checked only for zero (underflow guard on
/// `linear_conv_kernel_dim - 1`); it directly scales the GatedDeltaNet conv1d buffer.
#[test]
fn parser_rejects_linear_conv_kernel_dim_over_max() {
    let json = format!(
        r#"{{"text_config": {{"linear_conv_kernel_dim": {}}}}}"#,
        MAX_CONV_KERNEL_DIM + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("linear_conv_kernel_dim above MAX_CONV_KERNEL_DIM must be rejected")
        .to_string();
    assert!(
        err.contains("linear_conv_kernel_dim") && err.contains("MAX_CONV_KERNEL_DIM"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_linear_conv_kernel_dim_at_max() {
    let json = format!(r#"{{"text_config": {{"linear_conv_kernel_dim": {MAX_CONV_KERNEL_DIM}}}}}"#);
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "linear_conv_kernel_dim == MAX_CONV_KERNEL_DIM must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// GatedDeltaNet conv buffer two-factor product budget (MAX_GDN_CONV_BUFFER_SIZE)
// ──────────────────────────────────────────────────────────────────────

/// `MAX_LINEAR_NUM_KEY_HEADS`, `MAX_CONV_KERNEL_DIM`, and `MAX_GDN_STATE_SIZE` each bound
/// their own factor and are individually satisfied here (`linear_num_key_heads` and
/// `linear_conv_kernel_dim` sit exactly at their caps; `linear_num_value_heads *
/// linear_key_head_dim * linear_value_head_dim` sits exactly at `MAX_GDN_STATE_SIZE`), yet
/// the conv buffer element count (`linear_qkv_dim() * (linear_conv_kernel_dim - 1)`) still
/// exceeds `MAX_GDN_CONV_BUFFER_SIZE` by roughly three orders of magnitude -- demonstrating
/// the product-hostility the per-factor caps above do not close on their own.
#[test]
fn parser_rejects_gdn_conv_buffer_size_over_max_with_bounded_factors() {
    // `linear_num_key_heads` / `linear_num_value_heads` sit at their own caps and
    // `linear_key_head_dim` / `linear_value_head_dim` are kept small enough that neither
    // `MAX_GDN_STATE_SIZE` nor `MAX_GDN_CHUNK_SCRATCH_BYTES` fires first -- only the
    // `linear_conv_kernel_dim` product (`linear_qkv_dim() * (kernel_dim - 1)`) is over
    // `MAX_GDN_CONV_BUFFER_SIZE`.
    let json = format!(
        r#"{{"text_config": {{
                "linear_num_key_heads": {MAX_LINEAR_NUM_KEY_HEADS},
                "linear_num_value_heads": {MAX_LINEAR_NUM_VALUE_HEADS},
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": {MAX_CONV_KERNEL_DIM}
            }}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "a conv buffer product over MAX_GDN_CONV_BUFFER_SIZE must be rejected even \
                 though every individual factor is at or under its own cap",
        )
        .to_string();
    // CLASS A2 (round p): the aggregate MAX_GDN_SESSION_BYTES check runs earlier in
    // `validate()` than the MAX_GDN_CONV_BUFFER_SIZE check this test originally targeted,
    // and this same hostile geometry now also exceeds the aggregate budget -- so the
    // aggregate guard fires first. Both are legitimate rejections of the same hostile
    // input; accept either.
    assert!(
        (err.contains("GatedDeltaNet conv buffer size")
            && err.contains("MAX_GDN_CONV_BUFFER_SIZE"))
            || err.contains("MAX_GDN_SESSION_BYTES"),
        "wrong guard fired: {err}"
    );
}

/// Accept-case control: realistic GDN geometry (matching the preset) must still parse.
#[test]
fn parser_accepts_realistic_gdn_conv_buffer_size() {
    let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
        }}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "realistic GDN conv buffer geometry must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// RoPE table two-factor byte budget (MAX_ROPE_TABLE_BYTES)
// ──────────────────────────────────────────────────────────────────────

/// `MAX_POSITION_EMBEDDINGS` bounds `max_position_embeddings` and `MAX_HEAD_DIM` bounds
/// `rope_dim` transitively, both satisfied here (`max_position_embeddings` at its cap,
/// `head_dim` at its cap with `partial_rotary_factor = 1.0`), yet the RoPE table byte
/// count (`4 * max_position_embeddings * rope_dim`) exceeds `MAX_ROPE_TABLE_BYTES` by
/// roughly 32x -- demonstrating the product-hostility neither per-factor cap closes alone.
#[test]
fn parser_rejects_rope_table_bytes_over_max_with_bounded_factors() {
    let json = format!(
        r#"{{"text_config": {{
                "max_position_embeddings": {MAX_POSITION_EMBEDDINGS},
                "head_dim": {MAX_HEAD_DIM},
                "partial_rotary_factor": 1.0
            }}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "a RoPE table byte count over MAX_ROPE_TABLE_BYTES must be rejected even \
                 though max_position_embeddings and head_dim are each at or under their own cap",
        )
        .to_string();
    assert!(
        err.contains("RoPE table size") && err.contains("MAX_ROPE_TABLE_BYTES"),
        "wrong guard fired: {err}"
    );
}

/// Accept-case control: realistic RoPE geometry (matching a real Qwen3.5/3.6 preset) must
/// still parse.
#[test]
fn parser_accepts_realistic_rope_table_bytes() {
    let json = r#"{"text_config": {
            "max_position_embeddings": 262144,
            "head_dim": 256,
            "partial_rotary_factor": 0.25
        }}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "realistic RoPE table geometry must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// vision_config numeric field upper bounds
// ──────────────────────────────────────────────────────────────────────

/// Previously only overflow-guarded (via `checked_derived_sizes`); a huge-but-non-
/// overflowing `hidden_size` reaches those tensor-size derivations unchecked.
#[test]
fn parser_rejects_present_vision_config_with_hidden_size_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": {},
                "num_heads": 1,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
        MAX_VISION_HIDDEN_SIZE + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("hidden_size above MAX_VISION_HIDDEN_SIZE must be rejected")
        .to_string();
    assert!(
        err.contains("hidden_size") && err.contains("MAX_VISION_HIDDEN_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_hidden_size_at_max_over_byte_budget() {
    // CLASS A3 (round p): MAX_VISION_HIDDEN_SIZE alone is still generous (matching
    // MAX_HIDDEN_SIZE's scale), but composed into the merger fc1/fc2 and patch-embed
    // products, hidden_size at its own per-field cap now exceeds MAX_VISION_TENSOR_BYTES
    // even with every other field realistic -- the per-field cap is necessary but not
    // sufficient; the byte budget is the binding constraint. See MAX_VISION_TENSOR_BYTES
    // docs.
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": {MAX_VISION_HIDDEN_SIZE},
                "num_heads": 1,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("hidden_size == MAX_VISION_HIDDEN_SIZE must be rejected by the byte budget")
        .to_string();
    assert!(
        err.contains("MAX_VISION_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_num_heads_over_max() {
    let over = MAX_VISION_NUM_HEADS + 1;
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": {over},
                "num_heads": {over},
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("num_heads above MAX_VISION_NUM_HEADS must be rejected")
        .to_string();
    assert!(
        err.contains("num_heads") && err.contains("MAX_VISION_NUM_HEADS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_present_vision_config_with_num_heads_at_max() {
    // num_heads itself does not feed any `checked_derived_sizes` product (only
    // hidden_size % num_heads == 0 is checked), so it stays accepted at its own cap as
    // long as hidden_size (here set to a realistic divisible value, not also maxed) keeps
    // every derived tensor under MAX_VISION_TENSOR_BYTES. `in_channels` is 3 because the
    // preprocessor is RGB-only and every other value is now rejected outright; this test's
    // subject is `num_heads`, so the channel count is held at the one servable value.
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": {MAX_VISION_NUM_HEADS},
                "num_heads": {MAX_VISION_NUM_HEADS},
                "patch_size": 4,
                "spatial_merge_size": 1,
                "out_hidden_size": 1024,
                "temporal_patch_size": 1,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "num_heads == MAX_VISION_NUM_HEADS must be accepted"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_patch_size_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": {},
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
        MAX_VISION_PATCH_SIZE + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("patch_size above MAX_VISION_PATCH_SIZE must be rejected")
        .to_string();
    assert!(
        err.contains("patch_size") && err.contains("MAX_VISION_PATCH_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_patch_size_at_max_over_byte_budget() {
    // CLASS A3 (round p): patch_size is squared in patch_embed_weight; even at a
    // realistic hidden_size, patch_size at its own generous per-field cap overflows
    // MAX_VISION_TENSOR_BYTES. See MAX_VISION_TENSOR_BYTES docs.
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": {MAX_VISION_PATCH_SIZE},
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("patch_size == MAX_VISION_PATCH_SIZE must be rejected by the byte budget")
        .to_string();
    assert!(
        err.contains("MAX_VISION_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_spatial_merge_size_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": {},
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
        MAX_VISION_SPATIAL_MERGE_SIZE + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("spatial_merge_size above MAX_VISION_SPATIAL_MERGE_SIZE must be rejected")
        .to_string();
    assert!(
        err.contains("spatial_merge_size") && err.contains("MAX_VISION_SPATIAL_MERGE_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_spatial_merge_size_at_max_over_byte_budget() {
    // CLASS A3 (round p): spatial_merge_size is squared into merge_in, which is then
    // squared again for merger_fc1 -- a quartic blowup that overflows
    // MAX_VISION_TENSOR_BYTES at the field's own per-field cap even with everything else
    // realistic. See MAX_VISION_TENSOR_BYTES docs.
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": {MAX_VISION_SPATIAL_MERGE_SIZE},
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "spatial_merge_size == MAX_VISION_SPATIAL_MERGE_SIZE must be rejected by the \
                 byte budget",
        )
        .to_string();
    assert!(
        err.contains("MAX_VISION_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_out_hidden_size_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": {},
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
        MAX_VISION_OUT_HIDDEN_SIZE + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("out_hidden_size above MAX_VISION_OUT_HIDDEN_SIZE must be rejected")
        .to_string();
    assert!(
        err.contains("out_hidden_size") && err.contains("MAX_VISION_OUT_HIDDEN_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_out_hidden_size_at_max_over_byte_budget() {
    // CLASS A3 (round p): out_hidden_size is a direct factor of merger_fc2
    // (`out_hidden_size * merge_in`); at its own per-field cap it overflows
    // MAX_VISION_TENSOR_BYTES even with everything else realistic. See
    // MAX_VISION_TENSOR_BYTES docs.
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": {MAX_VISION_OUT_HIDDEN_SIZE},
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "out_hidden_size == MAX_VISION_OUT_HIDDEN_SIZE must be rejected by the byte \
                 budget",
        )
        .to_string();
    assert!(
        err.contains("MAX_VISION_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_temporal_patch_size_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": {},
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
        MAX_VISION_TEMPORAL_PATCH_SIZE + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("temporal_patch_size above MAX_VISION_TEMPORAL_PATCH_SIZE must be rejected")
        .to_string();
    assert!(
        err.contains("temporal_patch_size") && err.contains("MAX_VISION_TEMPORAL_PATCH_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_temporal_patch_size_at_max_over_byte_budget() {
    // CLASS A3 (round p): temporal_patch_size is a direct factor of patch_embed_weight;
    // at its own per-field cap it overflows MAX_VISION_TENSOR_BYTES even with everything
    // else realistic. See MAX_VISION_TENSOR_BYTES docs.
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": {MAX_VISION_TEMPORAL_PATCH_SIZE},
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "temporal_patch_size == MAX_VISION_TEMPORAL_PATCH_SIZE must be rejected by the \
                 byte budget",
        )
        .to_string();
    assert!(
        err.contains("MAX_VISION_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_num_position_embeddings_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": {},
                "in_channels": 3
            }}"#,
        MAX_VISION_NUM_POSITION_EMBEDDINGS + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "num_position_embeddings above MAX_VISION_NUM_POSITION_EMBEDDINGS must be \
                 rejected",
        )
        .to_string();
    assert!(
        err.contains("num_position_embeddings")
            && err.contains("MAX_VISION_NUM_POSITION_EMBEDDINGS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_num_position_embeddings_at_max_over_byte_budget() {
    // CLASS A3 (round p): num_position_embeddings is a direct factor of pos_embed; at its
    // own per-field cap it overflows MAX_VISION_TENSOR_BYTES even with everything else
    // realistic. See MAX_VISION_TENSOR_BYTES docs.
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": {MAX_VISION_NUM_POSITION_EMBEDDINGS},
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "num_position_embeddings == MAX_VISION_NUM_POSITION_EMBEDDINGS must be \
                 rejected by the byte budget",
        )
        .to_string();
    assert!(
        err.contains("MAX_VISION_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_non_square_num_position_embeddings() {
    let json = config_json_with_vision(
        r#"{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2305,
                "in_channels": 3
            }"#,
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("non-square num_position_embeddings must be rejected")
        .to_string();
    assert!(
        err.contains("num_position_embeddings") && err.contains("perfect square"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_present_vision_config_with_square_num_position_embeddings() {
    let json = config_json_with_vision(
        r#"{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }"#,
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "square num_position_embeddings (48^2 = 2304) must be accepted"
    );
}

#[test]
fn parser_rejects_zero_max_position_embeddings() {
    let json = r#"{"text_config": {"max_position_embeddings": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("max_position_embeddings: 0 must yield an InferenceError, not panic")
        .to_string();
    assert!(
        err.contains("max_position_embeddings"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_max_position_embeddings_at_one() {
    let json = r#"{"text_config": {"max_position_embeddings": 1}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "max_position_embeddings: 1 must be accepted (minimal nonzero RoPE table)"
    );
}

#[test]
fn parser_rejects_zero_rope_theta() {
    let json = r#"{"text_config": {"rope_theta": 0.0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("rope_theta: 0.0 must yield an InferenceError, not NaN/inf propagation")
        .to_string();
    assert!(err.contains("rope_theta"), "wrong guard fired: {err}");
}

#[test]
fn parser_rejects_negative_rope_theta() {
    let json = r#"{"text_config": {"rope_theta": -1.0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("negative rope_theta must yield an InferenceError")
        .to_string();
    assert!(err.contains("rope_theta"), "wrong guard fired: {err}");
}

// Note: a dedicated "non-finite rope_theta via config.json" test is not constructible --
// JSON has no `NaN`/`Infinity` literals, and serde_json itself rejects an out-of-range
// numeric literal (e.g. `1e400`) as a parse error before `from_config_json_str`'s
// validation ever runs, so that vector is already fail-closed one layer up. The
// `is_finite()` half of the `rope_theta` guard below is defense-in-depth for callers
// that construct `Qwen35Config` directly (bypassing JSON parsing); the zero/negative
// tests above cover the vectors actually reachable through `config.json`.

#[test]
fn parser_accepts_realistic_rope_theta() {
    let json = r#"{"text_config": {"rope_theta": 10000000.0}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "realistic rope_theta must be accepted"
    );
}

// ── FIX 15: rms_norm_eps must be finite and positive ────────────────────────────

#[test]
fn parser_rejects_zero_rms_norm_eps() {
    let json = r#"{"text_config": {"rms_norm_eps": 0.0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("rms_norm_eps: 0.0 must yield an InferenceError, not NaN propagation")
        .to_string();
    assert!(err.contains("rms_norm_eps"), "wrong guard fired: {err}");
}

#[test]
fn parser_rejects_negative_rms_norm_eps() {
    let json = r#"{"text_config": {"rms_norm_eps": -1e-6}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("negative rms_norm_eps must yield an InferenceError")
        .to_string();
    assert!(err.contains("rms_norm_eps"), "wrong guard fired: {err}");
}

// Note: NaN/Infinity are not JSON literals (see the rope_theta note above for the same
// reason) -- exercise the `is_finite()` half by constructing `Qwen35Config` directly
// and calling `validate()`, matching the defense-in-depth precedent for `rope_theta`.

#[test]
fn parser_rejects_nan_rms_norm_eps() {
    let mut cfg = Qwen35Config::qwen35_2b();
    cfg.rms_norm_eps = f32::NAN;
    let err = cfg
        .validate()
        .expect_err("NaN rms_norm_eps must yield an InferenceError")
        .to_string();
    assert!(err.contains("rms_norm_eps"), "wrong guard fired: {err}");
}

#[test]
fn parser_rejects_infinite_rms_norm_eps() {
    let mut cfg = Qwen35Config::qwen35_2b();
    cfg.rms_norm_eps = f32::INFINITY;
    let err = cfg
        .validate()
        .expect_err("infinite rms_norm_eps must yield an InferenceError")
        .to_string();
    assert!(err.contains("rms_norm_eps"), "wrong guard fired: {err}");
}

#[test]
fn parser_accepts_realistic_rms_norm_eps() {
    let json = r#"{"text_config": {"rms_norm_eps": 0.000001}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "realistic rms_norm_eps (1e-6) must be accepted"
    );
}

#[test]
fn parser_rejects_linear_num_value_heads_over_max() {
    let value_heads = MAX_LINEAR_NUM_VALUE_HEADS + 1;
    let json = format!(
        r#"{{"text_config": {{
                "linear_num_key_heads": 1,
                "linear_num_value_heads": {value_heads},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("linear_num_value_heads above MAX_LINEAR_NUM_VALUE_HEADS must be rejected")
        .to_string();
    assert!(
        err.contains("linear_num_value_heads") && err.contains("MAX_LINEAR_NUM_VALUE_HEADS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_linear_num_value_heads_at_max() {
    let json = format!(
        r#"{{"text_config": {{
                "linear_num_key_heads": 1,
                "linear_num_value_heads": {MAX_LINEAR_NUM_VALUE_HEADS},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "linear_num_value_heads == MAX_LINEAR_NUM_VALUE_HEADS must be accepted"
    );
}

/// The specific attack this guard closes: `linear_value_head_dim = 1`
/// lets `value_heads` alone satisfy `MAX_LINEAR_OUTPUT_DIM` and `MAX_GDN_STATE_SIZE`
/// (both product budgets with `value_head_dim` as a factor) at values far above what
/// `MAX_LINEAR_NUM_VALUE_HEADS` alone would allow through those two checks -- but a
/// large, non-degenerate `linear_key_head_dim` still drives the GatedDeltaNet
/// chunk-scratch buffers (`new_session_inner`, `forward/metal_qwen35.rs`) to multi-GiB
/// scale. This must be caught by `MAX_GDN_CHUNK_SCRATCH_BYTES`, not by the two
/// pre-existing product budgets (which this config satisfies trivially).
#[test]
fn parser_rejects_gdn_chunk_scratch_over_max_with_value_heads_and_state_size_in_budget() {
    let huge_key_dim = 600_000;
    let json = format!(
        r#"{{"text_config": {{
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": {huge_key_dim},
            "linear_value_head_dim": 1
        }}}}"#
    );
    // Sanity: this config passes both pre-existing product budgets trivially
    // (value_heads = 1, value_dim = 1, so both products collapse to `huge_key_dim` / 1).
    assert!(
        huge_key_dim <= MAX_GDN_STATE_SIZE,
        "test fixture assumption: gdn_state_size must be within MAX_GDN_STATE_SIZE"
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "a GDN chunk-scratch product over MAX_GDN_CHUNK_SCRATCH_BYTES must be rejected \
                 even though linear_num_value_heads, MAX_LINEAR_OUTPUT_DIM, and \
                 MAX_GDN_STATE_SIZE all pass",
        )
        .to_string();
    assert!(
        err.contains("chunk-scratch") && err.contains("MAX_GDN_CHUNK_SCRATCH_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_realistic_gdn_chunk_scratch_geometry() {
    let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128
        }}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "realistic GDN chunk-scratch geometry must be accepted"
    );
}

/// Both `in_channels` boundary cases land on the RGB-only semantic bound rather than the
/// `MAX_VISION_IN_CHANNELS` allocation bound, because `preprocess_qwen35_image` indexes a
/// fixed 3-element `Rgb<u8>` and any other channel count panics the worker on the first
/// image. The allocation constant is still exercised here so the two bounds cannot drift
/// apart unnoticed: a value at the allocation limit must be rejected too, and rejected by
/// the semantic guard.
#[test]
fn parser_rejects_present_vision_config_with_in_channels_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": {}
            }}"#,
        MAX_VISION_IN_CHANNELS + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("in_channels above MAX_VISION_IN_CHANNELS must be rejected")
        .to_string();
    assert!(
        err.contains("in_channels must be 3"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_in_channels_at_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": {MAX_VISION_IN_CHANNELS}
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("in_channels == MAX_VISION_IN_CHANNELS is not RGB and must be rejected")
        .to_string();
    assert!(
        err.contains("in_channels must be 3"),
        "wrong guard fired: {err}"
    );
}

/// The channel count the preprocessor can actually serve. Without this positive case the
/// rejection tests above are satisfied by a validator that rejects every `in_channels`.
#[test]
fn parser_accepts_present_vision_config_with_in_channels_three() {
    let json = config_json_with_vision(
        r#"{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }"#,
    );
    let cfg =
        Qwen35Config::from_config_json_str(&json).expect("in_channels: 3 vision_config must parse");
    assert_eq!(
        cfg.vision_config
            .expect("vision_config present")
            .in_channels,
        3
    );
}

/// Every rejected value on both sides of 3, so the guard is pinned as an equality rather
/// than as a one-sided bound.
///
/// The cases matter individually. `4` is the panic this guard exists to prevent: it is
/// shape-consistent everywhere else, so before this validation it parsed cleanly and
/// failed only when a request reached `pixel[3]`. `0` was previously rejected by its own
/// `> 0` check, which the RGB bound subsumes, so it pins that the zero case is still
/// refused rather than falling through. `1` and `2` are the interior values that panic
/// nothing and are therefore the easiest to lose: `pixel[0]` and `pixel[1]` are in bounds
/// for a 3-element `Rgb<u8>`, so an under-wide `in_channels` silently produces a
/// channel-truncated tensor that disagrees with the checkpoint's `patch_embed` weights
/// instead of failing loudly.
///
/// Without `1` and `2` the suite is not mutation-sensitive: rewriting the predicate as
/// `self.in_channels == 0 || self.in_channels > 3` passes every other case in this file
/// while letting both slip through. That mutation was executed against this tree, not
/// reasoned about — it passed 5/5 before these cases existed.
#[test]
fn parser_rejects_present_vision_config_with_any_in_channels_but_three() {
    for in_channels in [0_usize, 1, 2, 4] {
        let json = config_json_with_vision(&format!(
            r#"{{
                    "depth": 4,
                    "hidden_size": 768,
                    "num_heads": 12,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "out_hidden_size": 1024,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 2304,
                    "in_channels": {in_channels}
                }}"#
        ));
        let err = match Qwen35Config::from_config_json_str(&json) {
            Ok(_) => {
                panic!("in_channels: {in_channels} vision_config must be rejected at parse time")
            }
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("in_channels must be 3"),
            "in_channels: {in_channels} -- wrong guard fired: {err}"
        );
    }
}

// ── CLASS A2: aggregate GDN session-buffer budget ───────────────────────────────

#[test]
fn parser_rejects_hostile_gdn_aggregate_value_heads_geometry() {
    // Hostile geometry: passes the per-buffer MAX_GDN_CHUNK_SCRATCH_BYTES
    // guard and every per-field/product guard individually, but the AGGREGATE across all
    // GDN session buffers reaches multi-GiB.
    let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 4096,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 32
        }}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("hostile aggregate GDN geometry (value_heads=4096) must be rejected")
        .to_string();
    assert!(
        err.contains("MAX_GDN_SESSION_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_rejects_hostile_gdn_aggregate_key_head_dim_geometry() {
    // Second hostile geometry: value_heads=1 keeps the three-factor
    // MAX_GDN_STATE_SIZE product and the per-buffer chunk-scratch guard both passing at
    // their boundary, but the free `linear_key_head_dim` multiplier still blows up the
    // aggregate session-buffer sum.
    let json = r#"{"text_config": {
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 524288,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 16
        }}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("hostile aggregate GDN geometry (key_head_dim=524288) must be rejected")
        .to_string();
    assert!(
        err.contains("MAX_GDN_SESSION_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_realistic_gdn_aggregate_geometry() {
    // qwen36_27b's real worst-case GDN geometry (value_heads=48, key/value head dims
    // 128) must stay well inside MAX_GDN_SESSION_BYTES.
    let cfg = Qwen35Config::qwen36_27b();
    assert!(
        cfg.validate().is_ok(),
        "realistic qwen36_27b GDN aggregate geometry must be accepted"
    );
}

// ── FIX 9+14: aggregate GDN cross-layer persistent-state budget ─────────────────

#[test]
fn parser_rejects_hostile_gdn_cross_layer_state_aggregate() {
    // Hostile geometry: every per-layer/per-buffer guard (MAX_GDN_STATE_SIZE,
    // MAX_GDN_CONV_BUFFER_SIZE, MAX_GDN_CHUNK_SCRATCH_BYTES, MAX_GDN_SESSION_BYTES)
    // passes individually, but 512 linear-attention layers each
    // allocating this per-layer state blow past MAX_GDN_CROSS_LAYER_STATE_BYTES --
    // `generation.rs` allocates one `GatedDeltaNetState` per linear layer up front,
    // before the first token.
    let json = r#"{"text_config": {
            "num_hidden_layers": 512,
            "full_attention_interval": 999999,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 700,
            "linear_conv_kernel_dim": 4
        }}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("hostile 512-linear-layer GDN cross-layer aggregate must be rejected")
        .to_string();
    assert!(
        err.contains("MAX_GDN_CROSS_LAYER_STATE_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_realistic_gdn_cross_layer_state_aggregate() {
    // qwen36_27b's real linear-layer count and per-layer GDN state must stay well
    // inside MAX_GDN_CROSS_LAYER_STATE_BYTES.
    let cfg = Qwen35Config::qwen36_27b();
    assert!(
        cfg.validate().is_ok(),
        "realistic qwen36_27b GDN cross-layer aggregate must be accepted"
    );
}

// ── CLASS A3: materialization-site byte budgets ─────────────────────────────────

#[test]
fn parser_rejects_hostile_embedding_product() {
    let json = format!(
        r#"{{"text_config": {{"vocab_size": 4000000, "hidden_size": {MAX_HIDDEN_SIZE}}}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("hostile embedding product must be rejected before materialization")
        .to_string();
    assert!(
        err.contains("MAX_EMBEDDING_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_realistic_embedding_product() {
    let cfg = Qwen35Config::qwen36_35b_a3b();
    assert!(
        cfg.validate().is_ok(),
        "realistic embedding product must be accepted"
    );
}

// ── FIX 8: dense-FFN gate/up/down product budget ────────────────────────────────

#[test]
fn parser_rejects_hostile_dense_ffn_product() {
    // hidden_size and intermediate_size both at their individual per-field caps
    // (MAX_HIDDEN_SIZE, MAX_INTERMEDIATE_SIZE) pass those guards independently; their
    // product (~4 TiB per dense FFN tensor) does not until MAX_DENSE_FFN_TENSOR_BYTES
    // exists. vocab_size=1 keeps MAX_EMBEDDING_BYTES trivially satisfied so that
    // guard doesn't fire first. The default preset (`text_config` omitted fields fall
    // back to `qwen36_35b_a3b`, a MoE config) is MoE, and the dense-FFN budget is
    // gated on `!is_moe()`, so the four MoE-indicator fields are explicitly nulled to
    // exercise the dense (non-MoE) load path this guard actually protects.
    let json = format!(
        r#"{{"text_config": {{"vocab_size": 1, "hidden_size": {MAX_HIDDEN_SIZE}, "intermediate_size": {MAX_INTERMEDIATE_SIZE}, "num_experts": null, "num_experts_per_tok": null, "moe_intermediate_size": null, "shared_expert_intermediate_size": null}}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("hostile dense FFN product must be rejected before materialization")
        .to_string();
    assert!(
        err.contains("MAX_DENSE_FFN_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_realistic_dense_ffn_product() {
    let cfg = Qwen35Config::qwen35_2b();
    assert!(
        cfg.validate().is_ok(),
        "realistic dense FFN product must be accepted"
    );
}

#[test]
fn parser_rejects_hostile_vision_derived_tensor_product() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 4,
                "hidden_size": {MAX_VISION_HIDDEN_SIZE},
                "num_heads": 1,
                "patch_size": 14,
                "spatial_merge_size": 1,
                "out_hidden_size": {MAX_VISION_OUT_HIDDEN_SIZE},
                "temporal_patch_size": 2,
                "num_position_embeddings": 16777216,
                "in_channels": 3
            }}"#
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("hostile vision derived-tensor product must be rejected")
        .to_string();
    assert!(
        err.contains("MAX_VISION_TENSOR_BYTES"),
        "wrong guard fired: {err}"
    );
}

// ── CLASS C: ValidatedQwen35Config ingress enumeration ──────────────────────────

#[test]
fn validated_config_rejects_hostile_directly_constructed_config() {
    // A raw Qwen35Config built directly (bypassing from_config_json_str entirely, e.g.
    // via serde deserializing just the inner struct) must still be rejected by the
    // checked TryFrom conversion -- device-free, no loader/allocation involved.
    let mut cfg = Qwen35Config::qwen35_2b();
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = Some(4096);
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 32;
    let result: Result<ValidatedQwen35Config, InferenceError> = cfg.try_into();
    assert!(
        result.is_err(),
        "hostile directly-constructed config must fail ValidatedQwen35Config::try_from"
    );
}

#[test]
fn validated_config_accepts_and_derefs_realistic_config() {
    let validated = Qwen35Config::qwen35_2b()
        .validate()
        .expect("realistic preset must validate");
    // Deref exposes read-only field access without unwrapping.
    assert_eq!(validated.hidden_size, 2048);
    let raw = validated.into_inner();
    assert_eq!(raw.hidden_size, 2048);
}

#[test]
fn from_config_json_str_validated_matches_from_config_json_str() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen36_config.json"
    ));
    let raw = Qwen35Config::from_config_json_str(json).expect("raw parse succeeds");
    let validated =
        Qwen35Config::from_config_json_str_validated(json).expect("validated parse succeeds");
    assert_eq!(raw.hidden_size, validated.hidden_size);
    assert_eq!(raw.num_hidden_layers, validated.num_hidden_layers);
}
