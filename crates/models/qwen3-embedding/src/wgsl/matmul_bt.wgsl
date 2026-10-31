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
