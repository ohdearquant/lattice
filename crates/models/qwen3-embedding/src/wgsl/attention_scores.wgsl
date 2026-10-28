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
