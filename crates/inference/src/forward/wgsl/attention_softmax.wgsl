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
            // floating-point arithmetic (#795): WGSL's Finite Math Assumption
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
