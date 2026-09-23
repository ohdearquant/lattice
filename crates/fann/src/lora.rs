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

/// One adapter's contribution to a single blended `(layer_idx, module)`
/// projection: everything [`plan_blend`] and [`plan_grouped`] need, and
/// nothing else.
///
/// Deliberately narrower than either caller's own per-layer type
/// (`lattice_inference::forward::metal_qwen35::LoraLayerData` also carries
/// the A/B tensors) — this leaf crate cannot depend on that type anyway (the
/// dependency direction runs the other way, `inference` depends on `fann`),
/// and a blend *plan* never touches a buffer.
#[derive(Debug, Clone, Copy)]
pub struct BlendProjection<'a> {
    /// Transformer layer index (0-based).
    pub layer_idx: usize,
    /// Projection module name (e.g. `"q_proj"`, `"o_proj"`).
    pub module: &'a str,
    /// This adapter's rank for this projection.
    pub rank: usize,
    /// Input dimension.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
}

/// The planned shape of one blended `(layer_idx, module)` projection: every
/// contributing adapter's rank summed, at the group's agreed `(d_in, d_out)`.
///
/// `module` borrows from the same `BlendProjection<'a>` inputs `plan_blend`
/// or `plan_grouped` was given, rather than cloning each group's module
/// name: every caller (`blend_lora_layer_data`, the adapter residency
/// registry's `publish`) has the underlying `String` alive for the plan's
/// entire lifetime, so this avoids one allocation per `(layer_idx, module)`
/// group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedProjection<'a> {
    /// Transformer layer index (0-based).
    pub layer_idx: usize,
    /// Projection module name.
    pub module: &'a str,
    /// Summed rank across every adapter contributing to this projection.
    pub rank_total: usize,
    /// Input dimension, agreed by every contributing adapter.
    pub d_in: usize,
    /// Output dimension, agreed by every contributing adapter.
    pub d_out: usize,
}

/// Pre-allocation planning for a LoRA blend, given projections already
/// grouped by `(layer_idx, module)`.
///
/// This is [`plan_blend`]'s planning core, split out so a caller that has
/// already grouped its projections for its own purposes —
/// `blend_lora_layer_data` (`lattice_inference::forward::metal_qwen35`)
/// groups its inputs by `(layer_idx, module)` to build the blended A/B
/// buffers regardless — can hand that grouping straight to the checks below
/// instead of paying for a second `HashMap` and per-group `Vec` just to
/// re-derive a grouping it already has. `plan_blend` itself is now a thin
/// wrapper: it groups a flat iterator of projections and calls this
/// function.
///
/// Bounds the aggregate blend size across every group against
/// [`MAX_BLEND_TOTAL_ELEMENTS`] (pass 1), then within each group requires
/// every entry to agree on `(d_in, d_out)` and bounds the summed rank
/// against [`MAX_BLEND_RANK_TOTAL`] (pass 2) — the same two passes, in the
/// same order, with the same error strings, as `plan_blend` ran directly
/// over its own grouping. It never sees an A/B buffer
/// (`check_buffer_lengths` stays with the caller that owns them) and never
/// sees a per-request mixture weight (`check_finite_weight` is about a
/// request's router output, not this residency-shaped question).
///
/// `groups` is walked twice — once (cloned) for pass 1, once (moved) for
/// pass 2 — and each group's own entries are walked twice within pass 2 (see
/// that pass's comment for why it isn't one combined loop), which is why
/// both `G` and `I` carry a `Clone` bound. For the iterator shapes every
/// current caller passes (a `Map` over a `Vec`'s or `HashMap`'s own
/// borrowing `Iter`, with non-capturing closures), cloning is a pointer
/// copy, not an allocation.
///
/// # Preconditions
///
/// The caller guarantees each `(layer_idx, module)` key appears in `groups`
/// at most once and that every group's entries are non-empty (a
/// `HashMap`-based grouping pass, as both `plan_blend` and
/// `blend_lora_layer_data` run, satisfies this by construction: a key only
/// exists in the map because at least one projection was pushed into it).
/// An empty group is treated as a caller bug reported through `Err`, naming
/// the offending `(layer_idx, module)` — never a panic, never an
/// out-of-bounds index into an empty group.
///
/// # Errors
///
/// Returns `Err` when:
/// - any group in `groups` is empty;
/// - the aggregate blend size across every group exceeds
///   `MAX_BLEND_TOTAL_ELEMENTS`;
/// - two entries in the same group disagree on `(d_in, d_out)`;
/// - the summed rank for one group exceeds `MAX_BLEND_RANK_TOTAL`;
/// - rank accumulation or a size product overflows `usize`.
pub fn plan_grouped<'a, G, I>(ctx: &str, groups: G) -> Result<Vec<PlannedProjection<'a>>, String>
where
    G: IntoIterator<Item = (usize, &'a str, I)> + Clone,
    I: IntoIterator<Item = BlendProjection<'a>> + Clone,
{
    // Pass 1: bound the TOTAL planned allocation across every group before
    // validating any individual group's dimensions -- an oversized
    // aggregate rejects before the per-group dims walk in pass 2, mirroring
    // the original two-pass order. Each group's entries are consumed
    // exactly once here (the first entry for `(d_in, d_out)`, then the rest
    // for the rank sum), so this pass needs no `Clone` of a group's own
    // entries -- only `groups.clone()` itself, to leave `groups` available
    // for pass 2 below.
    let mut planned_elems: usize = 0;
    for (layer_idx, module, entries) in groups.clone() {
        let mut iter = entries.into_iter();
        let first = iter.next().ok_or_else(|| {
            format!("{ctx}: layer {layer_idx} module '{module}' has an empty projection group")
        })?;
        let mut group_rank = accumulate_rank(0, first.rank, ctx)?;
        for entry in iter {
            group_rank = accumulate_rank(group_rank, entry.rank, ctx)?;
        }
        let group_elems =
            checked_group_elements(ctx, layer_idx, module, group_rank, first.d_in, first.d_out)?;
        planned_elems = accumulate_planned_elements(planned_elems, group_elems, ctx)?;
    }
    check_aggregate_elements_cap(planned_elems, ctx)?;

    // Pass 2: per group, require every entry to agree on `(d_in, d_out)`
    // *before* summing rank. Kept as two separate walks -- a dims-match walk
    // over the full group, then a fresh rank sum -- rather than merged into
    // one loop that checks dims and accumulates rank per entry: a merged
    // loop can have an earlier entry's `accumulate_rank` overflow before a
    // later entry's dims mismatch is ever reached, which would report the
    // overflow instead of the mismatch for that input. The original
    // (pre-split) code ran its whole dims-match loop to completion before
    // its rank-sum loop ever started, so a dims mismatch anywhere in the
    // group always won that race; two walks here preserve exactly that
    // order. This is the one place a group's entries need their own
    // `Clone`: the dims-match walk clones them, the rank-sum walk consumes
    // the original.
    let mut result = Vec::new();
    for (layer_idx, module, entries) in groups {
        let mut dims_iter = entries.clone().into_iter().enumerate();
        let (_, first) = dims_iter.next().ok_or_else(|| {
            format!("{ctx}: layer {layer_idx} module '{module}' has an empty projection group")
        })?;
        let d_in = first.d_in;
        let d_out = first.d_out;

        // Entry 0 trivially matches itself (`d_in == d_in`, `d_out ==
        // d_out`); `dims_iter` already consumed it above extracting `first`,
        // so this only re-checks entries 1..N -- the same entries whose
        // check could ever fail in the original all-entries-including-0 loop.
        for (idx, entry) in dims_iter {
            check_dims_match(
                ctx,
                layer_idx,
                module,
                d_in,
                d_out,
                idx,
                entry.d_in,
                entry.d_out,
            )?;
        }

        let mut rank_total: usize = 0;
        for entry in entries {
            rank_total = accumulate_rank(rank_total, entry.rank, ctx)?;
        }
        check_rank_total_cap(rank_total, ctx)?;

        result.push(PlannedProjection {
            layer_idx,
            module,
            rank_total,
            d_in,
            d_out,
        });
    }
    Ok(result)
}

/// Pre-allocation planning for a LoRA blend, from a flat, ungrouped iterator
/// of projections.
///
/// Groups `projections` by `(layer_idx, module)` and hands the grouping to
/// [`plan_grouped`], which runs the actual checks (see its own docs for the
/// full pass-by-pass description). Both the blend itself and the adapter
/// residency registry's state publication call this (issue #1735) — the ONE
/// copy of these checks, so a `GET /v1/lora` report of whether the resident
/// set can be blended and the blend a routed request actually runs cannot
/// disagree. `blend_lora_layer_data` (`lattice_inference::forward::metal_qwen35`)
/// already groups its inputs to build the blended A/B buffers regardless, so
/// it calls [`plan_grouped`] directly instead of this function, to avoid
/// grouping the same projections a second time.
///
/// `ctx` is reproduced verbatim in every returned message, exactly like
/// every other function in this module: callers pass their own name, so
/// this refactor leaves their error text unchanged.
///
/// # Errors
///
/// Returns `Err` when:
/// - the aggregate blend size across every projection exceeds
///   `MAX_BLEND_TOTAL_ELEMENTS`;
/// - two entries in the same `(layer_idx, module)` group disagree on
///   `(d_in, d_out)`;
/// - the summed rank for one `(layer_idx, module)` group exceeds
///   `MAX_BLEND_RANK_TOTAL`;
/// - rank accumulation or a size product overflows `usize`.
pub fn plan_blend<'a>(
    ctx: &str,
    projections: impl IntoIterator<Item = BlendProjection<'a>>,
) -> Result<Vec<PlannedProjection<'a>>, String> {
    use std::collections::HashMap;

    let mut grouped: HashMap<(usize, &'a str), Vec<BlendProjection<'a>>> = HashMap::new();
    for projection in projections {
        grouped
            .entry((projection.layer_idx, projection.module))
            .or_default()
            .push(projection);
    }

    // `.iter().copied()` hands each group's already-collected
    // `Vec<BlendProjection<'a>>` to `plan_grouped` without any further
    // allocation (`BlendProjection` is `Copy`).
    let groups = grouped
        .iter()
        .map(|(&(layer_idx, module), entries)| (layer_idx, module, entries.iter().copied()));
    plan_grouped(ctx, groups)
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

    /// A feasible set: two adapters contributing to the same projection,
    /// well under both caps and agreeing on shape.
    #[test]
    fn plan_blend_accepts_a_feasible_set() {
        let planned = plan_blend(
            "ctx",
            [
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: 8,
                    d_in: 4,
                    d_out: 4,
                },
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: 4,
                    d_in: 4,
                    d_out: 4,
                },
            ],
        )
        .expect("a feasible set must plan");
        assert_eq!(planned.len(), 1, "one projection group in, one plan out");
        assert_eq!(planned[0].layer_idx, 0);
        assert_eq!(planned[0].module, "q_proj");
        assert_eq!(planned[0].rank_total, 12);
        assert_eq!((planned[0].d_in, planned[0].d_out), (4, 4));
    }

    /// Refusal (1): the summed rank for one projection exceeds the cap.
    #[test]
    fn plan_blend_rejects_rank_over_budget() {
        let err = plan_blend(
            "ctx",
            [
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: MAX_BLEND_RANK_TOTAL,
                    d_in: 1,
                    d_out: 1,
                },
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: 1,
                    d_in: 1,
                    d_out: 1,
                },
            ],
        )
        .expect_err("summed rank exceeding the cap must refuse");
        assert!(err.contains("exceeds MAX_BLEND_RANK_TOTAL"), "got: {err}");

        // The must-pass control: exactly at the cap succeeds, so the arm
        // above is refusing the OVER-budget case and not every input.
        assert!(
            plan_blend(
                "ctx",
                [BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: MAX_BLEND_RANK_TOTAL,
                    d_in: 1,
                    d_out: 1,
                }],
            )
            .is_ok()
        );
    }

    /// Refusal (2): the aggregate blend size across every projection
    /// exceeds the cap, even though each individual projection is within
    /// its own per-group rank budget.
    #[test]
    fn plan_blend_rejects_aggregate_over_budget() {
        let rank = MAX_BLEND_RANK_TOTAL; // exactly at the per-group cap
        let d_in = 2048usize;
        let d_out = 2048usize;
        // 65 distinct (layer_idx, module) groups: 4096*(2048+2048)*65 =
        // 1,090,519,040 > MAX_BLEND_TOTAL_ELEMENTS (1<<30).
        let projections: Vec<BlendProjection<'_>> = (0..65usize)
            .map(|layer_idx| BlendProjection {
                layer_idx,
                module: "q_proj",
                rank,
                d_in,
                d_out,
            })
            .collect();
        let err = plan_blend("ctx", projections).expect_err("aggregate over-budget must refuse");
        assert!(
            err.contains("aggregate") || err.contains("MAX_BLEND_TOTAL_ELEMENTS"),
            "got: {err}"
        );
    }

    /// Refusal (3): two adapters disagree on a projection's input or output
    /// width.
    #[test]
    fn plan_blend_rejects_dimension_mismatch() {
        let err = plan_blend(
            "ctx",
            [
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: 4,
                    d_in: 4,
                    d_out: 4,
                },
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: 4,
                    d_in: 8,
                    d_out: 4,
                },
            ],
        )
        .expect_err("a d_in mismatch must refuse");
        assert!(err.contains("mismatched dimensions"), "got: {err}");
    }

    /// An empty input plans to an empty (trivially feasible) result, rather
    /// than refusing -- the "must not be empty" rule belongs to
    /// `blend_lora_layer_data`'s own calling contract (a zero-adapter
    /// mixture means "base model", handled before it ever reaches a plan),
    /// not to plan feasibility itself. This matters for the residency
    /// registry, which must report a zero-adapter resident set as
    /// blend-feasible rather than refusing.
    #[test]
    fn plan_blend_of_no_projections_is_trivially_feasible() {
        assert_eq!(plan_blend("ctx", []), Ok(Vec::new()));
    }

    /// `plan_grouped` over projections already grouped by `(layer_idx,
    /// module)` must plan the same set (up to group order) as `plan_blend`
    /// grouping the same projections itself, for a feasible multi-group
    /// input -- the whole point of splitting `plan_grouped` out is that a
    /// caller who already has the grouping gets an identical plan without
    /// paying for `plan_blend`'s own re-grouping.
    #[test]
    fn plan_grouped_matches_plan_blend_on_a_feasible_multi_group_set() {
        let groups: Vec<(usize, &str, Vec<BlendProjection<'_>>)> = vec![
            (
                0,
                "q_proj",
                vec![
                    BlendProjection {
                        layer_idx: 0,
                        module: "q_proj",
                        rank: 8,
                        d_in: 4,
                        d_out: 4,
                    },
                    BlendProjection {
                        layer_idx: 0,
                        module: "q_proj",
                        rank: 4,
                        d_in: 4,
                        d_out: 4,
                    },
                ],
            ),
            (
                1,
                "k_proj",
                vec![BlendProjection {
                    layer_idx: 1,
                    module: "k_proj",
                    rank: 2,
                    d_in: 6,
                    d_out: 6,
                }],
            ),
        ];
        let flattened: Vec<BlendProjection<'_>> = groups
            .iter()
            .flat_map(|(_, _, entries)| entries.iter().copied())
            .collect();

        let mut via_grouped =
            plan_grouped("ctx", groups).expect("a feasible pre-grouped set must plan");
        let mut via_blend = plan_blend("ctx", flattened).expect("the same set flattened must plan");

        via_grouped.sort_by_key(|p| (p.layer_idx, p.module));
        via_blend.sort_by_key(|p| (p.layer_idx, p.module));
        assert_eq!(via_grouped, via_blend);
    }

    /// `plan_grouped` must refuse with the exact same error string as
    /// `plan_blend` on each of the three refusal fixtures above, so a
    /// caller that switches from `plan_blend` to `plan_grouped` (as
    /// `blend_lora_layer_data` does) sees no change in its own error text.
    #[test]
    fn plan_grouped_matches_plan_blend_error_strings_on_refusals() {
        // Refusal (1): summed rank over budget.
        let rank_over_budget = vec![
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: MAX_BLEND_RANK_TOTAL,
                d_in: 1,
                d_out: 1,
            },
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 1,
                d_in: 1,
                d_out: 1,
            },
        ];
        let err_blend = plan_blend("ctx", rank_over_budget.clone())
            .expect_err("summed rank exceeding the cap must refuse");
        let err_grouped = plan_grouped("ctx", vec![(0usize, "q_proj", rank_over_budget)])
            .expect_err("summed rank exceeding the cap must refuse");
        assert_eq!(err_blend, err_grouped);

        // Refusal (2): aggregate over budget, one entry per group so each
        // group's `plan_grouped` list mirrors `plan_blend`'s own grouping.
        let rank = MAX_BLEND_RANK_TOTAL;
        let d_in = 2048usize;
        let d_out = 2048usize;
        let aggregate_over_budget: Vec<BlendProjection<'_>> = (0..65usize)
            .map(|layer_idx| BlendProjection {
                layer_idx,
                module: "q_proj",
                rank,
                d_in,
                d_out,
            })
            .collect();
        let err_blend = plan_blend("ctx", aggregate_over_budget.clone())
            .expect_err("aggregate over-budget must refuse");
        let aggregate_groups: Vec<(usize, &str, Vec<BlendProjection<'_>>)> = aggregate_over_budget
            .iter()
            .map(|p| (p.layer_idx, p.module, vec![*p]))
            .collect();
        let err_grouped =
            plan_grouped("ctx", aggregate_groups).expect_err("aggregate over-budget must refuse");
        assert_eq!(err_blend, err_grouped);

        // Refusal (3): dimension mismatch within one group.
        let dimension_mismatch = vec![
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 4,
                d_in: 4,
                d_out: 4,
            },
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 4,
                d_in: 8,
                d_out: 4,
            },
        ];
        let err_blend =
            plan_blend("ctx", dimension_mismatch.clone()).expect_err("a d_in mismatch must refuse");
        let err_grouped = plan_grouped("ctx", vec![(0usize, "q_proj", dimension_mismatch)])
            .expect_err("a d_in mismatch must refuse");
        assert_eq!(err_blend, err_grouped);
    }

    /// An empty group is a caller bug reported through `Err`, never a panic
    /// (there is no `entries[0]` to index into).
    #[test]
    fn plan_grouped_rejects_an_empty_group() {
        let groups: Vec<(usize, &str, Vec<BlendProjection<'_>>)> =
            vec![(0usize, "q_proj", Vec::new())];
        let err = plan_grouped("ctx", groups).expect_err("an empty group must refuse, not panic");
        assert!(err.contains("empty"), "got: {err}");
    }
}
