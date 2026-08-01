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
