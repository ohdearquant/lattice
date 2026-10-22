pub(crate) fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

pub(crate) fn uniform_signed(state: &mut u32) -> f32 {
    let x = xorshift32(state);
    let u = x as f32 / u32::MAX as f32;
    u * 2.0 - 1.0
}
