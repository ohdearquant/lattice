// Tiled GEMM: C[M,N] = A[M,K] @ B^T[N,K]
// B is stored row-major [N,K], transposed in the multiply.

struct Dims { M: u32, N: u32, K: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm_bt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let ty = lid.y;
    let tx = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + tx;
        let b_col = t * TILE + ty;

        if (row < dims.M && a_col < dims.K) {
            tileA[ty][tx] = A[row * dims.K + a_col];
        } else {
            tileA[ty][tx] = 0.0;
        }

        // B is [N,K] row-major. For B^T multiply, we read B[col, b_col].
        if (col < dims.N && b_col < dims.K) {
            tileB[ty][tx] = B[col * dims.K + b_col];
        } else {
            tileB[ty][tx] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}

// Non-transposed: C[M,N] = A[M,K] @ B[K,N]
@compute @workgroup_size(16, 16)
fn gemm_nn(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let ty = lid.y;
    let tx = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + tx;
        let b_row = t * TILE + ty;

        if (row < dims.M && a_col < dims.K) {
            tileA[ty][tx] = A[row * dims.K + a_col];
        } else {
            tileA[ty][tx] = 0.0;
        }

        if (b_row < dims.K && col < dims.N) {
            tileB[ty][tx] = B[b_row * dims.N + col];
        } else {
            tileB[ty][tx] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
