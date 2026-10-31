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
