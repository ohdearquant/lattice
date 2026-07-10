//! WGSL shader source constants for matmul, RMSNorm, copy, add, and related GPU kernels.
pub(super) const MATMUL_BT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

const TILE: u32 = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

fn pu(i: u32) -> u32 {
    return params[i];
}

@compute @workgroup_size(16, 16, 1)
fn gemm_bt(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = pu(0u);
    let n = pu(1u);
    let kdim = pu(2u);

    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    let num_tiles = (kdim + TILE - 1u) / TILE;
    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_col = t * TILE + lid.y;
        if (row < m && a_col < kdim) {
            tileA[lid.y][lid.x] = A[row * kdim + a_col];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }
        if (col < n && b_col < kdim) {
            tileB[lid.y][lid.x] = B[col * kdim + b_col];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < TILE; kk = kk + 1u) {
            acc = acc + tileA[lid.y][kk] * tileB[kk][lid.x];
        }
        workgroupBarrier();
    }

    if (row < m && col < n) {
        C[row * n + col] = acc;
    }
}
"#;

pub(super) const RMS_NORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> GAMMA: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

var<workgroup> scratch: array<f32, 256>;

fn pu(i: u32) -> u32 {
    return params[i];
}

fn pf(i: u32) -> f32 {
    return bitcast<f32>(params[i]);
}

@compute @workgroup_size(256, 1, 1)
fn rms_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row_len = pu(3u);
    let num_rows = pu(4u);
    let eps = pf(12u);
    let row = wid.x;
    if (row >= num_rows) {
        return;
    }

    let base = row * row_len;
    var sum_sq: f32 = 0.0;
    for (var i: u32 = lid.x; i < row_len; i = i + 256u) {
        let v = X[base + i];
        sum_sq = sum_sq + v * v;
    }
    scratch[lid.x] = sum_sq;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    let inv_rms = inverseSqrt(scratch[0] / f32(row_len) + eps);
    for (var i: u32 = lid.x; i < row_len; i = i + 256u) {
        X[base + i] = X[base + i] * inv_rms * GAMMA[i];
    }
}
"#;

pub(super) const COPY_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> DST: array<f32>;
@group(0) @binding(1) var<storage, read> SRC: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn copy_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params[10u];
    let idx = gid.x;
    if (idx < total) {
        DST[idx] = SRC[idx];
    }
}
"#;

pub(super) const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> DST: array<f32>;
@group(0) @binding(1) var<storage, read> SRC: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params[10u];
    let idx = gid.x;
    if (idx < total) {
        DST[idx] = DST[idx] + SRC[idx];
    }
}
"#;

pub(super) const SILU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params[10u];
    let idx = gid.x;
    if (idx < total) {
        let v = X[idx];
        X[idx] = v * (1.0 / (1.0 + exp(-v)));
    }
}
"#;

pub(super) const MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> DST: array<f32>;
@group(0) @binding(1) var<storage, read> SRC: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params[10u];
    let idx = gid.x;
    if (idx < total) {
        DST[idx] = DST[idx] * SRC[idx];
    }
}
"#;

pub(super) const ROPE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> COS: array<f32>;
@group(0) @binding(2) var<storage, read> SIN: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn rope_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let seq_len = params[13u];
    let num_heads = params[6u];
    let head_dim = params[5u];
    let half_dim = params[9u];

    let pair = gid.x;
    let pos = gid.y;
    let head = gid.z;
    if (pair >= half_dim || pos >= seq_len || head >= num_heads) {
        return;
    }

    let base = ((pos * num_heads) + head) * head_dim;
    let rope_idx = pos * half_dim + pair;
    let x0 = X[base + pair];
    let x1 = X[base + half_dim + pair];
    let c = COS[rope_idx];
    let s = SIN[rope_idx];
    X[base + pair] = x0 * c - x1 * s;
    X[base + half_dim + pair] = x0 * s + x1 * c;
}
"#;

pub(super) const ATTENTION_SCORES_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> SCORES: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

const TILE: u32 = 16u;
var<workgroup> tileQ: array<array<f32, 16>, 16>;
var<workgroup> tileK: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn attention_scores(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let seq_len = params[13u];
    let num_heads = params[6u];
    let num_kv_heads = params[7u];
    let head_dim = params[5u];
    let groups = params[8u];

    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    let head = wid.z;
    if (head >= num_heads) {
        return;
    }
    let kv_head = head / groups;
    let num_tiles = (head_dim + TILE - 1u) / TILE;
    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let qk = t * TILE + lid.x;
        let kk = t * TILE + lid.y;

        if (row < seq_len && qk < head_dim) {
            let q_idx = ((row * num_heads) + head) * head_dim + qk;
            tileQ[lid.y][lid.x] = Q[q_idx];
        } else {
            tileQ[lid.y][lid.x] = 0.0;
        }

        if (col < seq_len && kk < head_dim) {
            let k_idx = ((col * num_kv_heads) + kv_head) * head_dim + kk;
            tileK[lid.y][lid.x] = K[k_idx];
        } else {
            tileK[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + tileQ[lid.y][i] * tileK[i][lid.x];
        }
        workgroupBarrier();
    }

    if (row < seq_len && col < seq_len) {
        let out_idx = ((head * seq_len) + row) * seq_len + col;
        SCORES[out_idx] = acc;
    }
}
"#;

// Guarantee scope (codex round-1 medium #1 on #795): the fail-closed
// zero-row contract below has been verified on this repository's native
// Metal-backed WGPU adapter (`GpuModelState::new`'s
// `wgpu::PowerPreference::HighPerformance` request, backend `Backends::all()`
// resolving to Metal on macOS CI/dev hosts). It is NOT a WGSL-portable
// guarantee: WGSL's Finite Math Assumption
// (https://www.w3.org/TR/WGSL/#finite-math-assumption) permits an
// implementation to assume NaN and infinities are absent during shader
// execution, so a different backend (Vulkan/DX12/browser WebGPU) may
// legally replace a poisoned value with an indeterminate finite-looking one
// at any point after it participates in WGSL floating-point arithmetic,
// before `is_non_finite`'s bitcast ever inspects it. Reading the raw score
// before the `* scale` multiply (below) narrows the window but cannot close
// it for backends whose compiler chooses to optimize on the assumption
// upstream of that read. The claim this kernel makes is therefore: "fails
// closed on the tested native backend". The CPU parity / differential
// (`e2e-parity.yml`, HF-vs-lattice greedy-token) and CPU-side
// `attention::softmax_row` gates define the reference semantics for future
// backend-specific differential coverage; detecting a defeated guard on an
// untested WGPU backend requires adding that backend to a differential run —
// those gates do not execute this shader on Vulkan/DX12/browser WebGPU today.
pub(super) const ATTENTION_SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> SCORES: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

var<workgroup> max_scratch: array<f32, 128>;
var<workgroup> sum_scratch: array<f32, 128>;
var<workgroup> bad_scratch: array<u32, 128>;

fn pf(i: u32) -> f32 {
    return bitcast<f32>(params[i]);
}

// ADR-080 C1 fail-closed row contract: a score is "bad" (NaN or +/-infinite)
// exactly when its IEEE-754 exponent bits are all set (0x7f800000 mask) --
// true for NaN (nonzero mantissa) and +/-Inf (zero mantissa) alike. WGSL's
// `max()` silently drops a NaN operand (same `maxNum` semantics the CPU
// contract's `row_max_and_any_nan` exists to counter), so the row max alone
// cannot be trusted to surface a poisoned score; this scans explicitly.
fn is_non_finite(v: f32) -> bool {
    let bits = bitcast<u32>(v);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

@compute @workgroup_size(128, 1, 1)
fn attention_softmax(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let seq_len = params[13u];
    let num_heads = params[6u];
    let scale = pf(11u);
    let row = wid.x;
    let head = wid.y;
    if (row >= seq_len || head >= num_heads) {
        return;
    }

    let row_base = ((head * seq_len) + row) * seq_len;

    // Pre-exp scan: row max over valid (k <= row) scaled scores, ignoring a
    // NaN/+-inf score for the max itself, plus a separate fail-closed flag
    // for any such score in the valid range (mirrors
    // `attention::softmax_row::row_fails_closed_pre_exp`).
    var local_max = -3.40282347e+38;
    var local_bad: u32 = 0u;
    for (var k: u32 = lid.x; k < seq_len; k = k + 128u) {
        if (k <= row) {
            // Inspect the raw score BEFORE it participates in any WGSL
            // floating-point arithmetic (codex round-1 medium #1 on #795):
            // WGSL's Finite Math Assumption
            // (https://www.w3.org/TR/WGSL/#finite-math-assumption) permits an
            // implementation to assume NaN/infinities are absent during
            // shader execution, so a runtime expression that *would*
            // mathematically produce one is legally free to return an
            // indeterminate value instead once it has passed through an
            // arithmetic op. Reading `SCORES[row_base + k]` and testing it
            // via `is_non_finite` prior to the `* scale` multiply removes
            // that multiply from the poisoned value's path -- `scale` is a
            // host-supplied, already-finite scalar (`pf(11u)`), so deferring
            // the check past it bought nothing but one more arithmetic step
            // for an implementation to legally launder the bit pattern
            // through. This narrows, but does not eliminate, the WGSL
            // portability gap: see the guarantee-scope comment below.
            let raw = SCORES[row_base + k];
            if (is_non_finite(raw)) {
                local_bad = 1u;
            } else {
                let v = raw * scale;
                local_max = max(local_max, v);
            }
        }
    }
    max_scratch[lid.x] = local_max;
    bad_scratch[lid.x] = local_bad;
    workgroupBarrier();

    var stride: u32 = 64u;
    loop {
        if (lid.x < stride) {
            max_scratch[lid.x] = max(max_scratch[lid.x], max_scratch[lid.x + stride]);
            bad_scratch[lid.x] = bad_scratch[lid.x] | bad_scratch[lid.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    let max_val = max_scratch[0u];
    let row_fails_closed_pre_exp = (bad_scratch[0u] != 0u);

    if (row_fails_closed_pre_exp) {
        // Fail-closed by ASSIGNMENT (attention::softmax_row::finalize_row's
        // `row.fill(0.0)`): never compute exp()/sum on a row already known to
        // carry a NaN or +/-infinite score. A later multiply-through-zero on
        // a NaN numerator would not recover to zero under IEEE-754
        // (`NaN * 0.0 == NaN`), so this returns before any such multiply.
        for (var k: u32 = lid.x; k < seq_len; k = k + 128u) {
            SCORES[row_base + k] = 0.0;
        }
        return;
    }

    var local_sum: f32 = 0.0;
    for (var k: u32 = lid.x; k < seq_len; k = k + 128u) {
        if (k <= row) {
            let e = exp(SCORES[row_base + k] * scale - max_val);
            SCORES[row_base + k] = e;
            local_sum = local_sum + e;
        } else {
            SCORES[row_base + k] = 0.0;
        }
    }
    sum_scratch[lid.x] = local_sum;
    workgroupBarrier();

    stride = 64u;
    loop {
        if (lid.x < stride) {
            sum_scratch[lid.x] = sum_scratch[lid.x] + sum_scratch[lid.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    let sum_val = sum_scratch[0u];
    // Fail-closed finalize by ASSIGNMENT (mirrors
    // `attention::softmax_row::finalize_row`): a non-positive or non-finite
    // denominator zeroes the row directly. The previous
    // `1.0 / max(sum_val, 1e-20)` floor-clamp is removed -- it manufactured a
    // finite-looking reciprocal for a NaN `sum_val` (WGSL `max()` drops the
    // NaN operand) while the numerator lanes were already NaN, leaking NaN
    // into the rest of the network instead of failing the row closed (#790).
    if (is_non_finite(sum_val) || sum_val <= 0.0) {
        for (var k: u32 = lid.x; k < seq_len; k = k + 128u) {
            SCORES[row_base + k] = 0.0;
        }
    } else {
        let inv_sum = 1.0 / sum_val;
        for (var k: u32 = lid.x; k < seq_len; k = k + 128u) {
            if (k <= row) {
                SCORES[row_base + k] = SCORES[row_base + k] * inv_sum;
            }
        }
    }
}
"#;

pub(super) const ATTENTION_CONTEXT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> SCORES: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

const TILE: u32 = 16u;
var<workgroup> tileScores: array<array<f32, 16>, 16>;
var<workgroup> tileV: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn attention_context(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let seq_len = params[13u];
    let num_heads = params[6u];
    let num_kv_heads = params[7u];
    let head_dim = params[5u];
    let groups = params[8u];

    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    let head = wid.z;
    if (head >= num_heads) {
        return;
    }
    let kv_head = head / groups;
    let num_tiles = (seq_len + TILE - 1u) / TILE;
    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let score_k = t * TILE + lid.x;
        let v_k = t * TILE + lid.y;

        if (row < seq_len && score_k < seq_len) {
            let s_idx = ((head * seq_len) + row) * seq_len + score_k;
            tileScores[lid.y][lid.x] = SCORES[s_idx];
        } else {
            tileScores[lid.y][lid.x] = 0.0;
        }

        if (col < head_dim && v_k < seq_len) {
            let v_idx = ((v_k * num_kv_heads) + kv_head) * head_dim + col;
            tileV[lid.y][lid.x] = V[v_idx];
        } else {
            tileV[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + tileScores[lid.y][i] * tileV[i][lid.x];
        }
        workgroupBarrier();
    }

    if (row < seq_len && col < head_dim) {
        let out_stride = num_heads * head_dim;
        let out_idx = row * out_stride + head * head_dim + col;
        OUT[out_idx] = acc;
    }
}
"#;
