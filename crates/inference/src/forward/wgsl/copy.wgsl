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
