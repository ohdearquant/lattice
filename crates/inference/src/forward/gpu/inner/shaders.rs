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

pub(super) const ATTENTION_SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> SCORES: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

var<workgroup> max_scratch: array<f32, 128>;
var<workgroup> sum_scratch: array<f32, 128>;

fn pf(i: u32) -> f32 {
    return bitcast<f32>(params[i]);
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
    var local_max = -3.40282347e+38;
    for (var k: u32 = lid.x; k < seq_len; k = k + 128u) {
        if (k <= row) {
            local_max = max(local_max, SCORES[row_base + k] * scale);
        }
    }
    max_scratch[lid.x] = local_max;
    workgroupBarrier();

    var stride: u32 = 64u;
    loop {
        if (lid.x < stride) {
            max_scratch[lid.x] = max(max_scratch[lid.x], max_scratch[lid.x + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    let max_val = max_scratch[0u];
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

    let inv_sum = 1.0 / max(sum_scratch[0u], 1e-20);
    for (var k: u32 = lid.x; k < seq_len; k = k + 128u) {
        if (k <= row) {
            SCORES[row_base + k] = SCORES[row_base + k] * inv_sum;
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
