use super::*;

// ---------------------------------------------------------------
// Deterministic PRNG (xorshift64, matches gqa.rs / differential.rs)
// ---------------------------------------------------------------

fn det_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 7;
        state ^= state >> 9;
        state = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mantissa = ((state >> 41) as u32) & 0x007f_ffff;
        let x = f32::from_bits(0x3f80_0000 | mantissa) - 1.5;
        out.push(x);
    }
    out
}

// ---------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------

/// Minimal valid config: l=4, d=2, l'=4, n=3, w=4. Satisfies d|l and d|l'.
fn small_cfg() -> NsaConfig {
    NsaConfig {
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        compress_block: 4,
        compress_stride: 2,
        select_block: 4,
        num_selected: 3,
        window: 4,
    }
}

/// Construct NsaWeights for tests. `model_dim` is the gate projection input width.
fn make_weights(cfg: &NsaConfig, model_dim: usize, seed: u64) -> NsaWeights {
    let l = cfg.compress_block;
    let head_dim = cfg.head_dim;
    let phi_in = cfg.phi_in();
    let mut s = seed;
    let mut next = |n: usize| -> Vec<f32> {
        s = s.wrapping_add(0x1234_5678_9abc_def0);
        det_data(n, s)
    };
    NsaWeights {
        phi_k_w1: next(phi_in * phi_in),
        phi_k_b1: next(phi_in),
        phi_k_w2: next(head_dim * phi_in),
        phi_k_b2: next(head_dim),
        phi_v_w1: next(phi_in * phi_in),
        phi_v_b1: next(phi_in),
        phi_v_w2: next(head_dim * phi_in),
        phi_v_b2: next(head_dim),
        k_intrablock_pos: next(cfg.num_kv_heads * l * head_dim),
        v_intrablock_pos: next(cfg.num_kv_heads * l * head_dim),
        g_proj_w: next(3 * cfg.num_heads * model_dim),
        g_proj_b: next(3 * cfg.num_heads),
    }
}

fn run_nsa(cfg: &NsaConfig, seq_len: usize, seed: u64) -> Vec<f32> {
    let model_dim = cfg.q_dim(); // use q_dim as model_dim in tests
    let weights = make_weights(cfg, model_dim, seed);
    let q = det_data(seq_len * cfg.q_dim(), seed + 1);
    let q_rope = det_data(seq_len * cfg.q_dim(), seed + 2);
    let k_cmp = det_data(seq_len * cfg.kv_dim(), seed + 3);
    let k_slc = det_data(seq_len * cfg.kv_dim(), seed + 4);
    let k_win = det_data(seq_len * cfg.kv_dim(), seed + 5);
    let v_cmp = det_data(seq_len * cfg.kv_dim(), seed + 6);
    let v_slc = det_data(seq_len * cfg.kv_dim(), seed + 7);
    let v_win = det_data(seq_len * cfg.kv_dim(), seed + 8);
    let x = det_data(seq_len * model_dim, seed + 9);
    let mut out = vec![0.0f32; seq_len * cfg.q_dim()];
    let mut scratch = NsaScratch::default();
    apply_native_sparse_attention(
        &q,
        &q_rope,
        &k_cmp,
        &k_slc,
        &k_win,
        &v_cmp,
        &v_slc,
        &v_win,
        &x,
        &weights,
        &mut out,
        seq_len,
        cfg,
        &mut scratch,
    );
    out
}

// ---------------------------------------------------------------
// Config validation guards
// ---------------------------------------------------------------

#[test]
fn test_nsa_config_validate_ok() {
    small_cfg().validate(); // must not panic
}

#[test]
#[should_panic(expected = "num_heads must be > 0")]
fn test_zero_num_heads_panics() {
    let mut cfg = small_cfg();
    cfg.num_heads = 0;
    cfg.validate();
}

#[test]
#[should_panic(expected = "num_kv_heads must be > 0")]
fn test_zero_num_kv_heads_panics() {
    let mut cfg = small_cfg();
    cfg.num_kv_heads = 0;
    cfg.validate();
}

#[test]
#[should_panic(expected = "head_dim must be > 0")]
fn test_zero_head_dim_panics() {
    let mut cfg = small_cfg();
    cfg.head_dim = 0;
    cfg.validate();
}

#[test]
#[should_panic(expected = "compress_block (l) must be > 0")]
fn test_zero_compress_block_panics() {
    let mut cfg = small_cfg();
    cfg.compress_block = 0;
    cfg.validate();
}

#[test]
#[should_panic(expected = "compress_stride (d) must be > 0")]
fn test_zero_compress_stride_panics() {
    let mut cfg = small_cfg();
    cfg.compress_stride = 0;
    cfg.validate();
}

#[test]
#[should_panic(expected = "num_heads (2) must be divisible by num_kv_heads (3)")]
fn test_non_divisible_heads_panics() {
    let mut cfg = small_cfg();
    cfg.num_kv_heads = 3; // 2 % 3 != 0
    cfg.validate();
}

#[test]
#[should_panic(expected = "compress_block (l=5) must be divisible by compress_stride (d=2)")]
fn test_l_not_divisible_by_d_panics() {
    let mut cfg = small_cfg();
    cfg.compress_block = 5; // 5 % 2 != 0
    cfg.validate();
}

#[test]
#[should_panic(expected = "select_block (l'=5) must be divisible by compress_stride (d=2)")]
fn test_lp_not_divisible_by_d_panics() {
    let mut cfg = small_cfg();
    cfg.select_block = 5; // 5 % 2 != 0
    cfg.validate();
}

#[test]
#[should_panic(expected = "num_selected (n=2) must be >= 3")]
fn test_num_selected_below_3_panics() {
    let mut cfg = small_cfg();
    cfg.num_selected = 2; // forced-block scheme needs 1 initial + 2 local
    cfg.validate();
}

#[test]
#[should_panic(expected = "compress_stride (d) must be > 0")]
fn test_zero_stride_via_apply_panics() {
    let cfg = NsaConfig {
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        compress_block: 4,
        compress_stride: 0, // invalid — must panic
        select_block: 4,
        num_selected: 3,
        window: 4,
    };
    // cfg.validate() runs first in apply_native_sparse_attention and panics on
    // compress_stride == 0 before any buffer or weight is inspected, so empty
    // values suffice here.
    let weights = NsaWeights {
        phi_k_w1: vec![],
        phi_k_b1: vec![],
        phi_k_w2: vec![],
        phi_k_b2: vec![],
        phi_v_w1: vec![],
        phi_v_b1: vec![],
        phi_v_w2: vec![],
        phi_v_b2: vec![],
        k_intrablock_pos: vec![],
        v_intrablock_pos: vec![],
        g_proj_w: vec![],
        g_proj_b: vec![],
    };
    let mut out: Vec<f32> = vec![];
    let mut scratch = NsaScratch::default();
    apply_native_sparse_attention(
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &weights,
        &mut out,
        1,
        &cfg,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "x_buf must be non-empty when seq_len > 0")]
fn test_empty_x_buf_nonempty_seq_panics() {
    // An empty x_buf for a non-empty sequence must be rejected, not silently
    // accepted as model_dim = 0 (which degrades the gate to bias-only).
    let cfg = small_cfg();
    let seq_len = 2;
    let weights = make_weights(&cfg, cfg.q_dim(), 1);
    let q = det_data(seq_len * cfg.q_dim(), 1);
    let q_rope = det_data(seq_len * cfg.q_dim(), 2);
    let k_cmp = det_data(seq_len * cfg.kv_dim(), 3);
    let k_slc = det_data(seq_len * cfg.kv_dim(), 4);
    let k_win = det_data(seq_len * cfg.kv_dim(), 5);
    let v_cmp = det_data(seq_len * cfg.kv_dim(), 6);
    let v_slc = det_data(seq_len * cfg.kv_dim(), 7);
    let v_win = det_data(seq_len * cfg.kv_dim(), 8);
    let x: Vec<f32> = vec![]; // empty x_buf with seq_len > 0 — must panic
    let mut out = vec![0.0f32; seq_len * cfg.q_dim()];
    let mut scratch = NsaScratch::default();
    apply_native_sparse_attention(
        &q,
        &q_rope,
        &k_cmp,
        &k_slc,
        &k_win,
        &v_cmp,
        &v_slc,
        &v_win,
        &x,
        &weights,
        &mut out,
        seq_len,
        &cfg,
        &mut scratch,
    );
}

// ---------------------------------------------------------------
// Shape tests
// ---------------------------------------------------------------

#[test]
fn test_nsa_shapes_seq1() {
    let cfg = small_cfg();
    let out = run_nsa(&cfg, 1, 42);
    assert_eq!(out.len(), cfg.q_dim());
}

#[test]
fn test_nsa_shapes_seq8() {
    let cfg = small_cfg();
    let out = run_nsa(&cfg, 8, 43);
    assert_eq!(out.len(), 8 * cfg.q_dim());
}

#[test]
fn test_nsa_shapes_seq16() {
    let cfg = small_cfg();
    let out = run_nsa(&cfg, 16, 44);
    assert_eq!(out.len(), 16 * cfg.q_dim());
}

// ---------------------------------------------------------------
// Finiteness
// ---------------------------------------------------------------

#[test]
fn test_nsa_output_finite_small() {
    let cfg = small_cfg();
    let out = run_nsa(&cfg, 12, 100);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}

#[test]
fn test_nsa_output_finite_seq1() {
    let cfg = small_cfg();
    let out = run_nsa(&cfg, 1, 101);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}

// ---------------------------------------------------------------
// Empty sequence
// ---------------------------------------------------------------

#[test]
fn test_nsa_seq_zero() {
    let cfg = small_cfg();
    let model_dim = cfg.q_dim();
    let weights = make_weights(&cfg, model_dim, 1);
    let mut out: Vec<f32> = vec![];
    let mut scratch = NsaScratch::default();
    // Must not panic; x_buf is empty (model_dim inferred as 0 for seq_len=0).
    apply_native_sparse_attention(
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &weights,
        &mut out,
        0,
        &cfg,
        &mut scratch,
    );
}

// ---------------------------------------------------------------
// Causal masking differential test
//
// Position 0 attends only to position 0 itself. Perturbing all future
// K/V (positions 1..seq_len-1) must leave position 0's output bit-identical.
// ---------------------------------------------------------------

#[test]
fn test_nsa_causal_masking() {
    let cfg = small_cfg();
    let seq_len = 10usize;
    let model_dim = cfg.q_dim();
    let weights = make_weights(&cfg, model_dim, 200);

    let q = det_data(seq_len * cfg.q_dim(), 201);
    let q_rope = det_data(seq_len * cfg.q_dim(), 202);
    let x = det_data(seq_len * model_dim, 203);
    // Six independent K/V branch buffers: k_cmp, k_slc, k_win, v_cmp, v_slc, v_win.
    let kv_base: [Vec<f32>; 6] =
        std::array::from_fn(|i| det_data(seq_len * cfg.kv_dim(), 210 + i as u64));

    // Perturb every K/V buffer at all positions except 0.
    let kv_perturbed: [Vec<f32>; 6] = std::array::from_fn(|i| {
        let mut b = kv_base[i].clone();
        for pos in 1..seq_len {
            for d in 0..cfg.kv_dim() {
                b[pos * cfg.kv_dim() + d] += 99_999.0;
            }
        }
        b
    });

    let run = |kv: &[Vec<f32>; 6]| {
        let mut out = vec![0.0f32; seq_len * cfg.q_dim()];
        let mut scratch = NsaScratch::default();
        apply_native_sparse_attention(
            &q,
            &q_rope,
            &kv[0],
            &kv[1],
            &kv[2],
            &kv[3],
            &kv[4],
            &kv[5],
            &x,
            &weights,
            &mut out,
            seq_len,
            &cfg,
            &mut scratch,
        );
        out
    };

    let out_base = run(&kv_base);
    let out_perturbed = run(&kv_perturbed);

    // Position 0's output must be bit-identical.
    for d in 0..cfg.q_dim() {
        assert_eq!(
            out_base[d].to_bits(),
            out_perturbed[d].to_bits(),
            "pos 0 changed when only future K/V changed — causal mask leak at dim {d}"
        );
    }

    // Sanity: at least one later position should differ after the perturbation.
    let any_later_changed = (cfg.q_dim()..seq_len * cfg.q_dim())
        .any(|i| out_base[i].to_bits() != out_perturbed[i].to_bits());
    assert!(
        any_later_changed,
        "no later position changed after perturbing future K/V — perturbation is a no-op"
    );
}

// ---------------------------------------------------------------
// Early-token behavior: seq_len < compress_block → zero compression blocks
// ---------------------------------------------------------------

#[test]
fn test_nsa_early_tokens_finite() {
    // seq_len=3 < compress_block=4 → no compression blocks ever causally valid.
    let cfg = small_cfg();
    let out = run_nsa(&cfg, 3, 300);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "output[{i}] = {v} not finite for early-token sequence"
        );
    }
}

// ---------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------

#[test]
fn test_nsa_deterministic() {
    let cfg = small_cfg();
    let out1 = run_nsa(&cfg, 8, 400);
    let out2 = run_nsa(&cfg, 8, 400);
    for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "output[{i}] differs between identical runs: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------
// GQA groups: multiple Q heads per KV head
// ---------------------------------------------------------------

#[test]
fn test_nsa_gqa_two_groups() {
    let cfg = NsaConfig {
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        compress_block: 4,
        compress_stride: 2,
        select_block: 4,
        num_selected: 3,
        window: 4,
    };
    let out = run_nsa(&cfg, 10, 500);
    assert_eq!(out.len(), 10 * cfg.q_dim());
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "GQA output[{i}] not finite: {v}");
    }
}

// ---------------------------------------------------------------
// Window-gate dominance test
//
// When g_cmp ≈ 0 and g_slc ≈ 0, the output should match
// g_win * (pure sliding-window output). Zero g_proj_w and set bias
// to [−large, −large, +large] per head.
// ---------------------------------------------------------------

#[test]
fn test_nsa_window_only_gate() {
    let cfg = small_cfg();
    let seq_len = 6usize;
    let model_dim = cfg.q_dim();
    let mut weights = make_weights(&cfg, model_dim, 600);

    weights.g_proj_w.fill(0.0);
    for h in 0..cfg.num_heads {
        weights.g_proj_b[3 * h] = -100.0; // g_cmp ≈ 0
        weights.g_proj_b[3 * h + 1] = -100.0; // g_slc ≈ 0
        weights.g_proj_b[3 * h + 2] = 100.0; // g_win ≈ 1
    }

    let q = det_data(seq_len * cfg.q_dim(), 601);
    let q_rope = det_data(seq_len * cfg.q_dim(), 602);
    let k_cmp = det_data(seq_len * cfg.kv_dim(), 603);
    let k_slc = det_data(seq_len * cfg.kv_dim(), 604);
    let k_win = det_data(seq_len * cfg.kv_dim(), 605);
    let v_cmp = det_data(seq_len * cfg.kv_dim(), 606);
    let v_slc = det_data(seq_len * cfg.kv_dim(), 607);
    let v_win = det_data(seq_len * cfg.kv_dim(), 608);
    let x = det_data(seq_len * model_dim, 609);

    let mut out_nsa = vec![0.0f32; seq_len * cfg.q_dim()];
    let mut scratch = NsaScratch::default();
    apply_native_sparse_attention(
        &q,
        &q_rope,
        &k_cmp,
        &k_slc,
        &k_win,
        &v_cmp,
        &v_slc,
        &v_win,
        &x,
        &weights,
        &mut out_nsa,
        seq_len,
        &cfg,
        &mut scratch,
    );

    // Pure sliding-window reference scaled by g_win — window branch uses k_win/v_win.
    let mut out_win = vec![0.0f32; seq_len * cfg.q_dim()];
    compute_sliding_window_reference(&q_rope, &k_win, &v_win, &mut out_win, seq_len, &cfg);
    let g_win = sigmoid(100.0_f32);
    for v in out_win.iter_mut() {
        *v *= g_win;
    }

    // g_cmp and g_slc = sigmoid(-100) ≈ 3.7e-44; their contribution is negligible.
    // Allow 1e-4 absolute tolerance.
    for (i, (&a, &b)) in out_nsa.iter().zip(out_win.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "window-only gate: output[{i}] mismatch: nsa={a} win_ref={b} diff={diff}"
        );
    }
}

fn compute_sliding_window_reference(
    q_rope: &[f32],
    k_rope: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    cfg: &NsaConfig,
) {
    let head_dim = cfg.head_dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let n_rep = cfg.n_rep();
    let scale = (head_dim as f32).powf(-0.5);
    let win = cfg.window;

    out.fill(0.0);
    for qt in 0..seq_len {
        let win_start = qt.saturating_sub(win - 1);
        let win_len = qt - win_start + 1;
        for kv_h in 0..cfg.num_kv_heads {
            for qh_local in 0..n_rep {
                let qh = kv_h * n_rep + qh_local;
                let q_off = qt * q_dim + qh * head_dim;
                let q_head = &q_rope[q_off..q_off + head_dim];
                let mut scores = vec![0.0f32; win_len];
                for (wi, tok) in (win_start..=qt).enumerate() {
                    let k_off = tok * kv_dim + kv_h * head_dim;
                    let dot: f32 = q_head
                        .iter()
                        .zip(k_rope[k_off..k_off + head_dim].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores[wi] = dot * scale;
                }
                softmax_inplace(&mut scores);
                let out_off = qt * q_dim + qh * head_dim;
                for (wi, tok) in (win_start..=qt).enumerate() {
                    let v_off = tok * kv_dim + kv_h * head_dim;
                    for dd in 0..head_dim {
                        out[out_off + dd] += scores[wi] * v[v_off + dd];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------
// Helper method correctness
// ---------------------------------------------------------------

#[test]
fn test_num_compress_blocks() {
    let cfg = small_cfg(); // l=4, d=2
    assert_eq!(cfg.num_compress_blocks(0), 0);
    assert_eq!(cfg.num_compress_blocks(3), 0);
    assert_eq!(cfg.num_compress_blocks(4), 1); // (4-4)/2+1=1
    assert_eq!(cfg.num_compress_blocks(5), 1); // (5-4)/2+1=1 (floor(1/2)=0)
    assert_eq!(cfg.num_compress_blocks(6), 2); // (6-4)/2+1=2
    assert_eq!(cfg.num_compress_blocks(10), 4); // (10-4)/2+1=4
}

#[test]
fn test_num_select_blocks() {
    let cfg = small_cfg(); // l'=4
    // ceil(seq_len / l') — a partial trailing block still counts (prefix-invariance).
    assert_eq!(cfg.num_select_blocks(0), 0);
    assert_eq!(cfg.num_select_blocks(3), 1); // ceil(3/4) = 1, not 0
    assert_eq!(cfg.num_select_blocks(4), 1);
    assert_eq!(cfg.num_select_blocks(7), 2); // ceil(7/4) = 2, not 1
    assert_eq!(cfg.num_select_blocks(8), 2);
}

#[test]
fn test_count_valid_compress_blocks() {
    // l=4, d=2, max_cblocks=5. Block i valid iff all tokens <= qt, i.e. i*d+l <= qt+1.
    // qt=2: 2+1=3 < 4 → 0 (block 0 covers [0,4), last token 3 > 2)
    assert_eq!(count_valid_compress_blocks(2, 4, 2, 5), 0);
    // qt=3: block 0 last token 3 <= 3 → 1
    assert_eq!(count_valid_compress_blocks(3, 4, 2, 5), 1);
    // qt=4: (4+1-4)/2+1 = 1 (block 1 covers [2,6), last token 5 > 4)
    assert_eq!(count_valid_compress_blocks(4, 4, 2, 5), 1);
    // qt=5: (5+1-4)/2+1 = 2 (blocks 0,1 valid; block 2 [4,8) last token 7 > 5)
    assert_eq!(count_valid_compress_blocks(5, 4, 2, 5), 2);
    // qt=6: (6+1-4)/2+1 = 2
    assert_eq!(count_valid_compress_blocks(6, 4, 2, 5), 2);
    // max clamp: qt=100 → min((97/2+1)=49, 5)=5
    assert_eq!(count_valid_compress_blocks(100, 4, 2, 5), 5);
}

#[test]
fn test_count_valid_select_blocks() {
    // l'=4, max_sblocks=5. Block j valid iff >= 1 token <= qt, i.e. j*l' <= qt.
    // qt=0: block 0 covers [0,4), token 0 <= 0 → 1
    assert_eq!(count_valid_select_blocks(0, 4, 5), 1);
    // qt=3: block 0 valid (0 <= 3); block 1 [4,8) has no token <= 3 → 1
    assert_eq!(count_valid_select_blocks(3, 4, 5), 1);
    // qt=4: block 1 now has token 4 <= 4 → 2
    assert_eq!(count_valid_select_blocks(4, 4, 5), 2);
    // qt=8: blocks 0,1,2 valid (block 2 [8,12) has token 8 <= 8) → 3
    assert_eq!(count_valid_select_blocks(8, 4, 5), 3);
    // Clamped by max_sblocks
    assert_eq!(count_valid_select_blocks(100, 4, 5), 5);
}

// ---------------------------------------------------------------
// Eq. 9 importance aggregation — hand-computed from the paper
//
// The expected sums below are worked out by hand from arXiv:2502.11089
// Eq. 9 (`p_cmp[(l'/d)*j - m - n]`). This check is grounded in the paper,
// NOT transcribed from the kernel's per-token loop or the test oracle —
// so it catches a spec-level misreading that a kernel-vs-oracle parity
// test (which shares one spec) cannot. See ADR-042 §KDC 5.
// ---------------------------------------------------------------

#[test]
fn test_aggregate_selection_importance_hand_computed() {
    // Powers of two, so any subset-sum is unambiguous: [1,2,4,...,256].
    let p: Vec<f32> = (0..9).map(|i| (1u32 << i) as f32).collect();

    // --- Non-default ratio: l=4, d=2, l'=8 → cps = l'/d = 4, ipc = l/d = 2 ---
    // j=0: ci = 0 - m - n; only (m,n)=(0,0) is in range → p[0].
    assert_eq!(aggregate_selection_importance(&p, 0, 4, 2), 1.0);
    // j=1: ci = 4 - m - n over m∈[0,4), n∈[0,2) → multiset {4,3,3,2,2,1,1,0}
    //      = p[0] + 2p[1] + 2p[2] + 2p[3] + p[4] = 1 + 4 + 8 + 16 + 16 = 45.
    assert_eq!(aggregate_selection_importance(&p, 1, 4, 2), 45.0);
    // j=2: ci = 8 - m - n → multiset {8,7,7,6,6,5,5,4}
    //      = p[4] + 2p[5] + 2p[6] + 2p[7] + p[8] = 16 + 64 + 128 + 256 + 256 = 720.
    assert_eq!(aggregate_selection_importance(&p, 2, 4, 2), 720.0);

    // --- Default ratio: l=4, d=2, l'=4 → cps = 2, ipc = 2 ---
    // j=1: ci = 2 - m - n over m,n∈[0,2) → multiset {2,1,1,0}
    //      = p[0] + 2p[1] + p[2] = 1 + 4 + 4 = 9.
    assert_eq!(aggregate_selection_importance(&p, 1, 2, 2), 9.0);

    // --- Upper-bound clamp: compression indices >= p_cmp.len() contribute 0 ---
    // j=2, cps=4, ipc=2 with p_cmp truncated to len 5: of the multiset
    // {8,7,7,6,6,5,5,4}, only index 4 is in range → p[4] = 16.
    assert_eq!(aggregate_selection_importance(&p[..5], 2, 4, 2), 16.0);

    // The old buggy `+ m + n` form would sum indices {4,5,5,6,6,7,7,8} for
    // j=1/cps=4/ipc=2 → 720, not 45. The `+`/`-` forms are disjoint here, so
    // the j=1 == 45 assertion above pins the sign.
}

// ---------------------------------------------------------------
// softmax_inplace unit tests
// ---------------------------------------------------------------

#[test]
fn test_softmax_inplace_sums_to_one() {
    let mut x = vec![1.0f32, 2.0, 3.0, 0.0, -1.0];
    softmax_inplace(&mut x);
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax sum should be 1.0, got {sum}"
    );
    for &v in &x {
        assert!(
            v >= 0.0 && v.is_finite(),
            "softmax output must be non-negative finite"
        );
    }
}

#[test]
fn test_softmax_inplace_single_element() {
    let mut x = vec![42.0f32];
    softmax_inplace(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-7);
}

#[test]
fn test_softmax_inplace_empty() {
    let mut x: Vec<f32> = vec![];
    softmax_inplace(&mut x); // must not panic
}

#[test]
fn test_softmax_preserves_relative_order() {
    let mut x = vec![3.0f32, 1.0, 2.0];
    let orig = x.clone();
    softmax_inplace(&mut x);
    // softmax is monotone: larger input → larger output
    assert!(x[0] > x[2], "softmax({}) > softmax({})", orig[0], orig[2]);
    assert!(x[2] > x[1], "softmax({}) > softmax({})", orig[2], orig[1]);
}

// ADR-080 C1: a NaN score must zero the WHOLE row, not leave NaN/Inf in
// place. RED before the fix: `if sum > 0.0` had no `else`, so `x` was left
// holding the raw (NaN-poisoned) unnormalized exponentials (#780).
#[test]
fn test_softmax_inplace_nan_fails_closed() {
    let mut x = vec![1.0f32, f32::NAN, 2.0, 0.5];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn test_softmax_inplace_pos_inf_fails_closed() {
    let mut x = vec![1.0f32, f32::INFINITY, 2.0];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn test_softmax_inplace_all_neg_inf_fails_closed() {
    let mut x = vec![f32::NEG_INFINITY; 3];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

// ---------------------------------------------------------------
// sigmoid unit tests
// ---------------------------------------------------------------

#[test]
fn test_sigmoid_values() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
    assert!(sigmoid(100.0) > 0.999);
    assert!(sigmoid(-100.0) < 0.001);
    // sigmoid is strictly monotone
    assert!(sigmoid(1.0) > sigmoid(0.0));
    assert!(sigmoid(-1.0) < sigmoid(0.0));
}

// ---------------------------------------------------------------
// Smoke test: many sequence lengths
// ---------------------------------------------------------------

#[test]
fn test_nsa_no_panic_various_seq_lens() {
    let cfg = small_cfg();
    for seq_len in [1, 2, 4, 5, 7, 8, 12, 16, 20] {
        let out = run_nsa(&cfg, seq_len, seq_len as u64 + 700);
        assert_eq!(
            out.len(),
            seq_len * cfg.q_dim(),
            "shape mismatch at seq_len={seq_len}"
        );
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "seq_len={seq_len}: output[{i}] not finite: {v}"
            );
        }
    }
}
