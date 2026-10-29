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
