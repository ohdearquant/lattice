use super::*;

/// The Qwen3.5-0.8B text decoder's projection shapes as `(name, n, k)`: one linear-attention
/// layer, the dense FFN and one full-attention layer.
const SHAPES: &[(&str, usize, usize)] = &[
    ("gdn.in_proj_qkv", 6144, 1024),
    ("gdn.in_proj_z", 2048, 1024),
    ("gdn.out_proj", 1024, 2048),
    ("ffn.gate", 3584, 1024),
    ("ffn.up", 3584, 1024),
    ("ffn.down", 1024, 3584),
    ("attn.q_gate", 4096, 1024),
    ("attn.k", 512, 1024),
    ("attn.o", 1024, 2048),
];

const ROWS: [usize; 3] = [1, 64, 512];

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn weight_bits(seed: u64, len: usize) -> Vec<u16> {
    let mut g = Lcg(seed);
    (0..len)
        .map(|_| (((g.unit() - 0.5) * 0.08).to_bits() >> 16) as u16)
        .collect()
}

fn activations(seed: u64, len: usize) -> Vec<f32> {
    let mut g = Lcg(seed);
    (0..len).map(|_| (g.unit() - 0.5) * 2.0).collect()
}

fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn l2_rows(x: &[f32], rows: usize, k: usize) -> Vec<f64> {
    (0..rows)
        .map(|r| {
            x[r * k..(r + 1) * k]
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// Runs one product both ways and asserts the f32 forward-error bound between them: each
/// result lies within `gamma_k * sum|a_i b_i|` of the exact dot product whatever the
/// summation order, and `sum|a_i b_i| <= |a| |b|`, so two of them differ by at most
/// `2 * gamma_k * |a| |b|`. Returns how many elements are bit-identical, an observation.
fn assert_within_f32_bound(name: &str, seed: u64, m: usize, k: usize, n: usize) -> usize {
    let bits = weight_bits(seed, n * k);
    let widened: Vec<f32> = bits.iter().map(|&b| widen(b)).collect();
    let a = activations(seed ^ 0x5eed, m * k);

    let mut want = vec![0.0f32; m * n];
    matmul_bt(&a, &widened, &mut want, m, k, n);
    let mut got = vec![f32::NAN; m * n];
    let mut scratch = Bf16MatmulScratch::new();
    matmul_bt_bf16(&a, &bits, &mut got, m, k, n, &mut scratch);

    let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
    let ku = k as f64 * unit_roundoff;
    let gamma = ku / (1.0 - ku);
    let a_norms = l2_rows(&a, m, k);
    let b_norms = l2_rows(&widened, n, k);

    let mut identical = 0;
    for r in 0..m {
        for j in 0..n {
            let (g, w) = (got[r * n + j], want[r * n + j]);
            let bound = 2.0 * gamma * a_norms[r] * b_norms[j];
            let diff = (f64::from(g) - f64::from(w)).abs();
            assert!(
                g.is_finite() && diff <= bound,
                "{name} m={m} n={n} k={k} row {r} col {j}: bf16 {g} vs f32 {w}, |diff| {diff:e} > bound {bound:e}"
            );
            if g.to_bits() == w.to_bits() {
                identical += 1;
            }
        }
    }
    identical
}

#[test]
fn matmul_bt_bf16_stays_within_the_f32_error_bound_on_the_qwen35_0_8b_shapes() {
    for (i, &(name, n, k)) in SHAPES.iter().enumerate() {
        for &m in &ROWS {
            let identical = assert_within_f32_bound(name, 100 + i as u64, m, k, n);
            eprintln!(
                "{name} m={m}: {identical} of {} elements bit-identical to matmul_bt",
                m * n
            );
        }
    }
}

#[test]
#[ignore = "allocates about 1.5 GB for the 248,320 x 1,024 tied embedding used as the logits matrix"]
fn matmul_bt_bf16_stays_within_the_f32_error_bound_on_the_logits_shape() {
    let (m, k, n) = (1, 1024, 248_320);
    let identical = assert_within_f32_bound("lm_head", 200, m, k, n);
    eprintln!("lm_head m={m}: {identical} of {n} elements bit-identical to matmul_bt");
}

#[test]
fn matmul_bt_bf16_widens_bf16_bits_exactly() {
    let values = [1.0f32, -2.0, 0.5, f32::MIN_POSITIVE, 3.0, -0.0];
    let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let mut c = vec![f32::NAN; values.len()];
    let mut scratch = Bf16MatmulScratch::new();
    matmul_bt_bf16(&[1.0], &bits, &mut c, 1, 1, values.len(), &mut scratch);
    for (got, want) in c.iter().zip(values) {
        assert_eq!(got.abs().to_bits(), want.abs().to_bits(), "{got} vs {want}");
    }
}

#[test]
fn matmul_bt_bf16_places_every_tile_in_its_own_columns() {
    // Three full tiles and a partial one. Row j of B is (j % 256, j / 256, 1, 0), all exact in
    // bf16, and the rows of A pick out its first three entries, so the first two output rows
    // name the B row each column came from.
    let (m, k) = (3, 4);
    let n = 3 * BF16_TILE_ROWS + 5;
    let a: Vec<f32> = (0..m * k)
        .map(|i| if i % k == i / k { 1.0 } else { 0.0 })
        .collect();
    let entry = |j: usize, col: usize| -> f32 {
        match col {
            0 => (j % 256) as f32,
            1 => (j / 256) as f32,
            2 => 1.0,
            _ => 0.0,
        }
    };
    let bits: Vec<u16> = (0..n * k)
        .map(|i| (entry(i / k, i % k).to_bits() >> 16) as u16)
        .collect();
    let mut c = vec![f32::NAN; m * n];
    let mut scratch = Bf16MatmulScratch::new();
    matmul_bt_bf16(&a, &bits, &mut c, m, k, n, &mut scratch);
    for r in 0..m {
        for j in 0..n {
            assert_eq!(c[r * n + j], entry(j, r), "row {r} col {j}");
        }
    }
}

#[test]
fn matmul_bt_bf16_reuses_a_scratch_sized_for_the_largest_product() {
    let (max_m, max_k) = (8, 64);
    let mut scratch = Bf16MatmulScratch::with_capacity(max_m, max_k);
    let held = scratch.held_elements();
    let (b_ptr, c_ptr) = (scratch.b_tile.as_ptr(), scratch.c_tile.as_ptr());
    for &(m, k, n) in &[
        (max_m, max_k, 2 * BF16_TILE_ROWS + 3),
        (1, max_k, 2 * BF16_TILE_ROWS + 3),
        (max_m, 32, 17),
        (max_m, max_k, 2 * BF16_TILE_ROWS + 3),
    ] {
        let bits = weight_bits(7, n * k);
        let a = activations(8, m * k);
        let mut c = vec![0.0f32; m * n];
        matmul_bt_bf16(&a, &bits, &mut c, m, k, n, &mut scratch);
    }
    assert_eq!(scratch.held_elements(), held);
    assert_eq!(scratch.b_tile.as_ptr(), b_ptr);
    assert_eq!(scratch.c_tile.as_ptr(), c_ptr);
}

#[test]
fn matmul_bt_bf16_zero_fills_for_an_empty_inner_dimension() {
    let mut c = vec![f32::NAN; 6];
    let mut scratch = Bf16MatmulScratch::new();
    matmul_bt_bf16(&[], &[], &mut c, 2, 0, 3, &mut scratch);
    assert!(c.iter().all(|&v| v == 0.0));
}

#[test]
fn matmul_bt_bf16_leaves_the_output_alone_for_no_rows_or_columns() {
    let mut c = vec![f32::NAN; 4];
    let mut scratch = Bf16MatmulScratch::new();
    matmul_bt_bf16(&[], &[0; 8], &mut c, 0, 4, 2, &mut scratch);
    matmul_bt_bf16(&[0.0; 8], &[], &mut c, 2, 4, 0, &mut scratch);
    assert!(c.iter().all(|v| v.is_nan()));
    assert_eq!(scratch.held_elements(), 0);
}

#[test]
#[should_panic(expected = "matmul_bt_bf16: b too short for n*k")]
fn matmul_bt_bf16_rejects_short_b() {
    let mut c = vec![0.0f32; 2];
    matmul_bt_bf16(
        &[0.0; 4],
        &[0; 7],
        &mut c,
        1,
        4,
        2,
        &mut Bf16MatmulScratch::new(),
    );
}

#[test]
#[should_panic(expected = "matmul_bt_bf16: shape overflow: n*k")]
fn matmul_bt_bf16_rejects_overflow() {
    let mut c = [0.0f32; 2];
    matmul_bt_bf16(
        &[0.0; 2],
        &[0; 2],
        &mut c,
        2,
        2,
        usize::MAX,
        &mut Bf16MatmulScratch::new(),
    );
}
