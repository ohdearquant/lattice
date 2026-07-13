#include <metal_stdlib>
using namespace metal;

// ===== Shared RMS-norm threadgroup reduction (#854) =====
// The barriered sum-of-squares reduction tree below — write to `shared[lid]`,
// barrier, tree-reduce in `tgs/2` steps, `rsqrt(mean + eps)` — was duplicated
// verbatim across eight Metal kernels (rms_norm, rms_norm_qwen35,
// per_head_rms_norm, fused_residual_add_norm, copy_and_rms_norm,
// copy_and_rms_norm_batch, fused_residual_add_norm_batch,
// per_head_rms_norm_batch). This is a mechanical extraction only: barrier
// count, tree order, and the final rsqrt expression are unchanged from every
// prior copy — RMS-norm is a nonlinearity and f32 reduction order is part of
// the numerical contract, so this helper must never reassociate the tree.
//
// Callers keep their own `local_sum` accumulation (and any input scan / copy
// / residual-add work fused into that phase), their own `threadgroup float
// shared[...]` array declaration, and their own epilogue gamma scaling
// (plain `gamma[i]` for `rms_norm`, Qwen3.5 shifted `(1 + gamma[i])`
// everywhere else) — this helper covers only the shared reduction, not the
// epilogue, since the epilogue differs per caller.
//
// Every thread in the threadgroup must call this on the same control-flow
// path (no call behind lane-local divergence) — the barriers inside require
// full threadgroup participation.
inline float rms_inv_from_local_sum(
    threadgroup float* shared,
    float local_sum,
    uint lid,
    uint tgs,
    uint width,
    float eps)
{
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return rsqrt(shared[0] / float(width) + eps);
}
