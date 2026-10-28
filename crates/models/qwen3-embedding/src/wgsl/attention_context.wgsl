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
