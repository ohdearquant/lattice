//! Shared LoRA adapter descriptor and blend-validation primitives.
//!
//! `lattice-tune` and `lattice-inference` each own an independent LoRA
//! adapter implementation (`lattice_tune::lora::LoraAdapter` for training and
//! CPU serving; `lattice_inference::forward::metal_qwen35` for the Metal GPU
//! decode path), and the crate dependency direction — `tune` may depend on
//! `inference`, never the reverse — stops either from importing the other's
//! validation rules directly. This module lives in `lattice-fann` because it
//! is the only leaf crate both sides already reach: `tune` depends on it
//! unconditionally, and `inference` depends on it optionally (the `mixture`
//! and `metal-gpu` features both enable it), so a rule fixed here is fixed
//! for every caller instead of drifting between two copies.
//!
//! Scope is deliberately narrow: adapter identity (rank, alpha, target
//! modules, dtype) and the pure, allocation-free checks that guard adapter
//! blending (weight finiteness, rank/element budget caps, buffer-shape
//! consistency). The actual blend math (concatenating and scaling A/B
//! buffers) and the forward-pass kernels stay in each crate — they operate
//! on crate-local types (`LoraLayer` vs `LoraLayerData`) and are not the
//! source of the drift this module fixes.

/// A LoRA adapter's static identity: rank, scaling factor, target modules,
/// and the tensor dtype it was trained/saved in.
///
/// `dtype` is a free-form label (e.g. `"f32"`, `"f16"`, `"bf16"`) — it is not
/// validated here, matching the existing manifest convention where real
/// tensor dtypes are checked independently at load time.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraDescriptor {
    /// Low-rank dimension. Typical values: 4, 8, 16, 32, 64.
    pub rank: usize,
    /// Scaling factor. The effective scale is `alpha / rank`.
    pub alpha: f32,
    /// Names of the modules that have LoRA adapters, e.g. `["q_proj", "v_proj"]`.
    pub target_modules: Vec<String>,
    /// Tensor dtype label the adapter was trained/saved in.
    pub dtype: String,
}

impl LoraDescriptor {
    /// Compute the LoRA scaling factor: `alpha / rank`.
    ///
    /// A zero rank has an effective scale of zero (an empty factorization
    /// contributes nothing), and a non-finite `alpha` or resulting scale
    /// also collapses to zero rather than propagating NaN/Inf into the
    /// forward pass.
    pub fn scale(&self) -> f32 {
        effective_scale(self.rank, self.alpha)
    }

    /// Validate that `alpha` and the effective `alpha / rank` scale are finite.
    pub fn validate(&self) -> Result<(), String> {
        validate_alpha_finite(self.rank, self.alpha)
    }
}

/// Recognized LoRA target-module names across every architecture this
/// project trains or serves adapters for: full-attention (GQA) `q_proj`,
/// `k_proj`, `v_proj`, `o_proj`; linear-attention (GDN) `in_proj_qkv`,
/// `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`; MLP `gate_proj`,
/// `up_proj`, `down_proj`; BERT `query`, `key`, `value`, `attn_output`,
/// `ffn_intermediate`, `ffn_output`.
///
/// This is a flat name allowlist, not an architecture-aware shape check —
/// whether a given module is valid for a *specific* layer's type (e.g. a GDN
/// module on a full-attention layer) is `qwen35_projection_shape`'s job in
/// `lattice-inference`, which both `lattice-tune` and the Metal load path
/// already call. This list exists so a descriptor's declared
/// `target_modules` can be checked for typos or unrecognized names before
/// any model architecture is known, in the one leaf crate both consumers share.
pub const KNOWN_LORA_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "query",
    "key",
    "value",
    "attn_output",
    "ffn_intermediate",
    "ffn_output",
];

/// Reject any `target_modules` entry that is not present in `known`.
pub fn validate_target_modules(target_modules: &[String], known: &[&str]) -> Result<(), String> {
    let unknown: Vec<&str> = target_modules
        .iter()
        .map(String::as_str)
        .filter(|m| !known.contains(m))
        .collect();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unknown LoRA target module(s): {}",
            unknown.join(", ")
        ))
    }
}

/// Compute `alpha / rank`, treating rank `0` and any non-finite result as `0.0`.
///
/// Free function form of [`LoraDescriptor::scale`] for callers that only
/// have `(rank, alpha)` on hand and do not want to build a full descriptor.
pub fn effective_scale(rank: usize, alpha: f32) -> f32 {
    let scale = if rank == 0 { 0.0 } else { alpha / rank as f32 };
    if alpha.is_finite() && scale.is_finite() {
        scale
    } else {
        0.0
    }
}

/// Validate that `alpha` and the effective `alpha / rank` scale are finite.
///
/// Free function form of [`LoraDescriptor::validate`].
pub fn validate_alpha_finite(rank: usize, alpha: f32) -> Result<(), String> {
    if !alpha.is_finite() {
        return Err(format!("LoRA alpha must be finite, got {alpha}"));
    }
    let scale = if rank == 0 { 0.0 } else { alpha / rank as f32 };
    if !scale.is_finite() {
        return Err(format!("LoRA effective scale must be finite, got {scale}"));
    }
    Ok(())
}

/// Maximum summed rank for one blended projection.
///
/// The Metal GEMV kernels assume a modest rank budget (≤ ~64 per adapter in
/// a typical mixture); this cap bounds allocations and rejects
/// adversarially large adapter pools before any `Vec::with_capacity`.
pub const MAX_BLEND_RANK_TOTAL: usize = 4096;

/// Aggregate cap on a blended adapter's total element count, summed across
/// every `(layer_idx, module)` projection: `Σ rank_total·(d_in + d_out)`.
/// At f32 this bounds the blended-adapter allocation to ~4 GiB.
pub const MAX_BLEND_TOTAL_ELEMENTS: usize = 1usize << 30; // 1,073,741,824 elements ≈ 4 GiB f32

/// Reject a non-finite blend mixture weight.
///
/// `ctx` is the caller's function name, reproduced verbatim in the error
/// message so each crate's existing message format is unchanged.
pub fn check_finite_weight(ctx: &str, idx: usize, weight: f32) -> Result<(), String> {
    if !weight.is_finite() {
        Err(format!(
            "{ctx}: weight at index {idx} is not finite ({weight})"
        ))
    } else {
        Ok(())
    }
}

/// Checked-accumulate `rank` into `acc`.
pub fn accumulate_rank(acc: usize, rank: usize, ctx: &str) -> Result<usize, String> {
    acc.checked_add(rank)
        .ok_or_else(|| format!("{ctx}: rank_total overflowed usize"))
}

/// Reject a summed rank exceeding [`MAX_BLEND_RANK_TOTAL`].
pub fn check_rank_total_cap(rank_total: usize, ctx: &str) -> Result<(), String> {
    if rank_total > MAX_BLEND_RANK_TOTAL {
        Err(format!(
            "{ctx}: summed rank {rank_total} exceeds MAX_BLEND_RANK_TOTAL={MAX_BLEND_RANK_TOTAL}"
        ))
    } else {
        Ok(())
    }
}

/// Reject mismatched `(d_in, d_out)` between the first entry of a projection
/// group and a later entry being folded into it.
#[allow(clippy::too_many_arguments)]
pub fn check_dims_match(
    ctx: &str,
    layer_idx: usize,
    module: &str,
    d_in: usize,
    d_out: usize,
    idx: usize,
    entry_d_in: usize,
    entry_d_out: usize,
) -> Result<(), String> {
    if entry_d_in != d_in || entry_d_out != d_out {
        Err(format!(
            "{ctx}: layer {layer_idx} module '{module}' has mismatched dimensions \
             (entry 0: d_in={d_in}, d_out={d_out}; entry {idx}: d_in={entry_d_in}, d_out={entry_d_out})"
        ))
    } else {
        Ok(())
    }
}

/// Verify an entry's A/B slice lengths match its declared `rank`, `d_in`,
/// and `d_out` (row-major `A: (rank, d_in)`, `B: (d_out, rank)`).
pub fn check_buffer_lengths(
    ctx: &str,
    idx: usize,
    rank: usize,
    d_in: usize,
    d_out: usize,
    a_len: usize,
    b_len: usize,
) -> Result<(), String> {
    let expected_a = rank
        .checked_mul(d_in)
        .ok_or_else(|| format!("{ctx}: rank*d_in overflowed usize"))?;
    let expected_b = d_out
        .checked_mul(rank)
        .ok_or_else(|| format!("{ctx}: d_out*rank overflowed usize"))?;
    if a_len != expected_a {
        return Err(format!(
            "{ctx}: entry {idx} A slice length {a_len} \
             does not match rank*d_in={rank}*{d_in}={expected_a}"
        ));
    }
    if b_len != expected_b {
        return Err(format!(
            "{ctx}: entry {idx} B slice length {b_len} \
             does not match d_out*rank={d_out}*{rank}={expected_b}"
        ));
    }
    Ok(())
}

/// Checked `rank_total * (d_in + d_out)` for one projection group, used to
/// build the aggregate element budget across every group before allocating.
pub fn checked_group_elements(
    ctx: &str,
    layer_idx: usize,
    module: &str,
    rank_total: usize,
    d_in: usize,
    d_out: usize,
) -> Result<usize, String> {
    let dims = d_in.checked_add(d_out).ok_or_else(|| {
        format!("{ctx}: layer {layer_idx} module '{module}' d_in+d_out overflowed usize")
    })?;
    rank_total
        .checked_mul(dims)
        .ok_or_else(|| format!("{ctx}: rank_total*(d_in+d_out) overflowed usize"))
}

/// Checked-accumulate one group's element count into the running aggregate.
pub fn accumulate_planned_elements(
    acc: usize,
    group_elems: usize,
    ctx: &str,
) -> Result<usize, String> {
    acc.checked_add(group_elems)
        .ok_or_else(|| format!("{ctx}: aggregate blend element count overflowed usize"))
}

/// Reject an aggregate element count exceeding [`MAX_BLEND_TOTAL_ELEMENTS`].
pub fn check_aggregate_elements_cap(planned_elems: usize, ctx: &str) -> Result<(), String> {
    if planned_elems > MAX_BLEND_TOTAL_ELEMENTS {
        // `MAX_BLEND_TOTAL_ELEMENTS * 4` is a compile-time constant expression:
        // on a 32-bit `usize` target (e.g. `wasm32-unknown-unknown`) computing
        // it in `usize` overflows u32::MAX and fails the build outright
        // (`#[deny(arithmetic_overflow)]`), not just at large runtime inputs.
        // Widening to `u64` first keeps the display math off the target's
        // native word size; `fann` is a public leaf crate other targets embed.
        let gib = (MAX_BLEND_TOTAL_ELEMENTS as u64 * 4) / (1024 * 1024 * 1024);
        Err(format!(
            "{ctx}: aggregate blend size {planned_elems} elements exceeds \
             MAX_BLEND_TOTAL_ELEMENTS={MAX_BLEND_TOTAL_ELEMENTS} (~{gib} GiB f32); reduce the \
             number of adapters, their rank, or the number of target projections",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_scale_zero_rank_is_zero() {
        assert_eq!(effective_scale(0, 4.0), 0.0);
    }

    #[test]
    fn effective_scale_matches_alpha_over_rank() {
        assert_eq!(effective_scale(2, 4.0), 2.0);
    }

    #[test]
    fn effective_scale_non_finite_alpha_collapses_to_zero() {
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(effective_scale(8, alpha), 0.0);
        }
    }

    #[test]
    fn validate_alpha_finite_rejects_non_finite() {
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_alpha_finite(8, alpha).unwrap_err();
            assert!(err.contains("alpha must be finite"));
        }
    }

    /// Direct regression for the `rank == 0` short-circuit: without it,
    /// `alpha / rank as f32` divides by zero and produces `+inf` for a
    /// positive finite `alpha`, which the very next check (`!scale.is_finite()`)
    /// would then reject — wrongly failing a legitimate zero-rank adapter
    /// (an empty factorization contributing nothing) that this branch exists
    /// to accept.
    #[test]
    fn validate_alpha_finite_accepts_zero_rank_with_finite_alpha() {
        assert!(validate_alpha_finite(0, 1.0).is_ok());
        assert_eq!(effective_scale(0, 1.0), 0.0);
    }

    /// Direct regression for the effective-scale finite guard itself
    /// (distinct from the `rank == 0` branch above): a typical finite
    /// `(rank, alpha)` pair must validate. Given `alpha` is already checked
    /// finite and `rank == 0` is short-circuited above this line, `alpha /
    /// rank as f32` for a nonzero `rank` cannot itself become non-finite —
    /// so this guard has no reachable "must fail" input today, and this test
    /// instead pins the guard's "must pass" side: inverting the
    /// `!scale.is_finite()` condition would make this fail directly, at the
    /// fann crate level, instead of only through the many downstream
    /// `lattice-tune` callers that route through it.
    #[test]
    fn validate_alpha_finite_accepts_typical_finite_scale() {
        assert!(validate_alpha_finite(8, 16.0).is_ok());
    }

    #[test]
    fn descriptor_validate_delegates_to_free_function() {
        let d = LoraDescriptor {
            rank: 8,
            alpha: f32::NAN,
            target_modules: vec![],
            dtype: "f32".into(),
        };
        assert!(d.validate().is_err());
        assert_eq!(d.scale(), 0.0);
    }

    #[test]
    fn check_finite_weight_rejects_nan() {
        let err = check_finite_weight("ctx", 3, f32::NAN).unwrap_err();
        assert!(err.contains("weight at index 3 is not finite"));
    }

    #[test]
    fn check_rank_total_cap_rejects_over_budget() {
        let err = check_rank_total_cap(MAX_BLEND_RANK_TOTAL + 1, "ctx").unwrap_err();
        assert!(err.contains("exceeds MAX_BLEND_RANK_TOTAL"));
        assert!(check_rank_total_cap(MAX_BLEND_RANK_TOTAL, "ctx").is_ok());
    }

    #[test]
    fn check_aggregate_elements_cap_rejects_over_budget() {
        let err = check_aggregate_elements_cap(MAX_BLEND_TOTAL_ELEMENTS + 1, "ctx").unwrap_err();
        assert!(err.contains("exceeds MAX_BLEND_TOTAL_ELEMENTS") || err.contains("aggregate"));
        assert!(
            err.contains("4 GiB"),
            "error must report the 4 GiB budget; got: {err}"
        );
        assert!(check_aggregate_elements_cap(MAX_BLEND_TOTAL_ELEMENTS, "ctx").is_ok());
    }

    /// `MAX_BLEND_TOTAL_ELEMENTS * 4` is a compile-time-constant expression:
    /// on a 32-bit `usize` target (`wasm32-unknown-unknown`), evaluating it
    /// in `usize` overflows u32::MAX and fails the build under
    /// `#[deny(arithmetic_overflow)]`, regardless of `planned_elems` at
    /// runtime — `cargo test` on this (64-bit) host cannot reproduce that,
    /// so this pins the u64-widened math's result directly as a same-crate
    /// regression guard; the 32-bit build itself was verified separately
    /// with `rustc --target wasm32-unknown-unknown`.
    #[test]
    fn max_blend_total_elements_times_four_survives_u64_widening() {
        let widened = (MAX_BLEND_TOTAL_ELEMENTS as u64).checked_mul(4);
        assert!(widened.is_some(), "widened GiB math must not overflow u64");
        assert_eq!(widened.unwrap() / (1024 * 1024 * 1024), 4);
    }

    #[test]
    fn check_buffer_lengths_rejects_short_a() {
        let err = check_buffer_lengths("ctx", 0, 2, 4, 4, 7, 8).unwrap_err();
        assert!(err.contains("A slice length"));
    }

    #[test]
    fn check_buffer_lengths_rejects_short_b() {
        let err = check_buffer_lengths("ctx", 0, 2, 4, 4, 8, 7).unwrap_err();
        assert!(err.contains("B slice length"));
    }

    /// Direct regression for the `rank.checked_mul(d_in)` overflow guard:
    /// replacing it with `unwrap_or(0)` would silently treat an overflowing
    /// `rank*d_in` as `0`, so any `a_len` would then satisfy `a_len ==
    /// expected_a` only when `a_len == 0` — a real overflow would either
    /// false-reject a correctly-sized (impossibly large) buffer or, worse,
    /// false-accept an empty one. Neither `checked_group_elements`'s own
    /// overflow tests above nor any existing `check_buffer_lengths` test
    /// drives `rank*d_in` past `usize::MAX`.
    #[test]
    fn check_buffer_lengths_rejects_rank_times_d_in_overflow() {
        let err = check_buffer_lengths("ctx", 0, usize::MAX, 2, 1, 0, 0).unwrap_err();
        assert!(
            err.contains("rank*d_in overflowed usize"),
            "expected rank*d_in overflow message; got: {err}"
        );
    }

    /// Mirror of the above for `d_out.checked_mul(rank)`. Both `expected_a`
    /// and `expected_b` are computed before either length is checked, so
    /// `rank*d_in` (`2*1=2`) must itself stay within bounds for this to
    /// reach and isolate the `d_out*rank` guard rather than failing on the
    /// earlier `rank*d_in` guard instead.
    #[test]
    fn check_buffer_lengths_rejects_d_out_times_rank_overflow() {
        let err = check_buffer_lengths("ctx", 0, 2, 1, usize::MAX, 0, 0).unwrap_err();
        assert!(
            err.contains("d_out*rank overflowed usize"),
            "expected d_out*rank overflow message; got: {err}"
        );
    }

    #[test]
    fn check_dims_match_rejects_mismatch() {
        let err = check_dims_match("ctx", 0, "q_proj", 4, 4, 1, 4, 8).unwrap_err();
        assert!(err.contains("mismatched dimensions"));
    }

    #[test]
    fn accumulate_rank_overflow_errors() {
        let err = accumulate_rank(usize::MAX, 1, "ctx").unwrap_err();
        assert!(err.contains("overflowed usize"));
    }

    #[test]
    fn checked_group_elements_rejects_dims_add_overflow() {
        let err = checked_group_elements("ctx", 0, "q_proj", 1, usize::MAX, 1).unwrap_err();
        assert!(
            err.contains("d_in+d_out overflowed"),
            "expected d_in+d_out overflow message; got: {err}"
        );
    }

    #[test]
    fn checked_group_elements_rejects_rank_dims_mul_overflow() {
        // dims = d_in + d_out = 2 (no overflow); rank_total * dims = usize::MAX * 2
        // overflows the multiplication.
        let err = checked_group_elements("ctx", 0, "q_proj", usize::MAX, 1, 1).unwrap_err();
        assert!(
            err.contains("rank_total*(d_in+d_out) overflowed"),
            "expected rank_total*(d_in+d_out) overflow message; got: {err}"
        );
    }

    #[test]
    fn accumulate_planned_elements_rejects_overflow() {
        let err = accumulate_planned_elements(usize::MAX, 1, "ctx").unwrap_err();
        assert!(
            err.contains("aggregate blend element count overflowed"),
            "expected aggregate overflow message; got: {err}"
        );
    }

    #[test]
    fn validate_target_modules_accepts_known_names() {
        let modules = vec!["q_proj".to_string(), "up_proj".to_string()];
        assert!(validate_target_modules(&modules, KNOWN_LORA_TARGET_MODULES).is_ok());
    }

    #[test]
    fn validate_target_modules_rejects_unknown_name() {
        let modules = vec!["q_proj".to_string(), "not_a_real_module".to_string()];
        let err = validate_target_modules(&modules, KNOWN_LORA_TARGET_MODULES).unwrap_err();
        assert!(err.contains("not_a_real_module"));
    }

    #[test]
    fn validate_target_modules_empty_is_ok() {
        assert!(validate_target_modules(&[], KNOWN_LORA_TARGET_MODULES).is_ok());
    }
}
