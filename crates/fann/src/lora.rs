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

/// Reject an entry whose own `(layer_idx, module)` disagrees with the group
/// key it was placed under.
///
/// `plan_grouped` takes its grouping as given: the key comes from the
/// `(usize, &str, I)` tuple a caller built, and each entry inside `I` also
/// carries its own `layer_idx`/`module`. Nothing before this check compared
/// the two, so a caller keeping them in sync by hand -- and every caller
/// does, since a group is normally built by pushing an entry into the map
/// slot for its own `(layer_idx, module)` -- had no instrument to catch a
/// slip. An entry filed under the wrong key blends silently into the wrong
/// projection's plan; this rejects that instead.
pub fn check_group_key_match(
    ctx: &str,
    layer_idx: usize,
    module: &str,
    idx: usize,
    entry_layer_idx: usize,
    entry_module: &str,
) -> Result<(), String> {
    if entry_layer_idx != layer_idx || entry_module != module {
        Err(format!(
            "{ctx}: layer {layer_idx} module '{module}' group has a key mismatch at entry {idx} \
             (entry is for layer {entry_layer_idx} module '{entry_module}')"
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
/// at most once (a `HashMap`-based grouping pass, as both `plan_blend` and
/// `blend_lora_layer_data` run, satisfies this by construction). Two other
/// properties a caller's own grouping is expected to hold -- that every
/// group is non-empty, and that every entry's own `layer_idx`/`module`
/// agrees with the key it was filed under -- are checked rather than
/// trusted: either is reported through `Err`, naming the offending
/// `(layer_idx, module)`, never a panic, never an out-of-bounds index into
/// an empty group, and never a plan built from an entry under the wrong key.
///
/// # Errors
///
/// Returns `Err` when:
/// - any group in `groups` is empty;
/// - an entry's own `(layer_idx, module)` disagrees with the group key it
///   was placed under;
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
        let mut iter = entries.into_iter().enumerate();
        let (_, first) = iter.next().ok_or_else(|| {
            format!("{ctx}: layer {layer_idx} module '{module}' has an empty projection group")
        })?;
        check_group_key_match(ctx, layer_idx, module, 0, first.layer_idx, first.module)?;
        let mut group_rank = accumulate_rank(0, first.rank, ctx)?;
        for (idx, entry) in iter {
            check_group_key_match(ctx, layer_idx, module, idx, entry.layer_idx, entry.module)?;
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
mod tests;
