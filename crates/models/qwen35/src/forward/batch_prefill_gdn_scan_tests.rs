use super::super::tests::tiny_test_config;
use super::*;
use crate::attention::gdn::require_qwen35_gdn_geometry;

fn noise(rng: &mut u64, len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            ((*rng >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn config(key_heads: usize, value_heads: usize, key_dim: usize, value_dim: usize) -> Qwen35Config {
    let mut cfg = tiny_test_config();
    cfg.linear_num_key_heads = key_heads;
    cfg.linear_num_value_heads = Some(value_heads);
    cfg.linear_key_head_dim = key_dim;
    cfg.linear_value_head_dim = value_dim;
    cfg.linear_conv_kernel_dim = 4;
    cfg
}

fn weights(cfg: &Qwen35Config, rng: &mut u64) -> GatedDeltaNetWeights {
    let qkv_dim = cfg.linear_qkv_dim();
    let value_heads = cfg.linear_num_value_heads();
    let kernel = cfg.linear_conv_kernel_dim;
    GatedDeltaNetWeights {
        in_proj_qkv: Vec::new().into(),
        in_proj_qkv_rows: 0,
        in_proj_qkv_cols: 0,
        in_proj_z: Vec::new().into(),
        in_proj_z_rows: 0,
        in_proj_z_cols: 0,
        in_proj_b: Vec::new(),
        in_proj_b_rows: 0,
        in_proj_b_cols: 0,
        in_proj_a: Vec::new(),
        in_proj_a_rows: 0,
        in_proj_a_cols: 0,
        a_log: noise(rng, value_heads, 0.5),
        dt_bias: noise(rng, value_heads, 0.5),
        conv1d_weight: noise(rng, qkv_dim * kernel, 0.5),
        conv_dim: qkv_dim,
        kernel_size: kernel,
        norm_weight: noise(rng, cfg.linear_output_dim(), 0.5),
        out_proj: Vec::new().into(),
        out_proj_rows: 0,
        out_proj_cols: 0,
    }
}

struct Workspace {
    conv_token: Vec<f32>,
    conv_history: Vec<f32>,
    conv_acc: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    kv_mem: Vec<f32>,
    delta: Vec<f32>,
    out_head: Vec<f32>,
}

impl Workspace {
    /// Sized as the prefill scratch sizes them, filled with NaN so a value read
    /// before it is written shows up in the output.
    fn new(cfg: &Qwen35Config) -> Self {
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let key_heads_dim = cfg.linear_num_value_heads() * cfg.linear_key_head_dim;
        let history = (CONV_BLOCKS + 1) * (cfg.linear_conv_kernel_dim - 1) * qkv_dim;
        Self {
            conv_token: vec![f32::NAN; qkv_dim],
            conv_history: vec![f32::NAN; history],
            conv_acc: vec![f32::NAN; CONV_BLOCKS * qkv_dim],
            q: vec![f32::NAN; key_heads_dim],
            k: vec![f32::NAN; key_heads_dim],
            kv_mem: vec![f32::NAN; output_dim],
            delta: vec![f32::NAN; output_dim],
            out_head: vec![f32::NAN; output_dim],
        }
    }

    fn work(&mut self) -> ScanWork<'_> {
        ScanWork {
            conv_token: &mut self.conv_token,
            conv_history: &mut self.conv_history,
            conv_acc: &mut self.conv_acc,
            q: &mut self.q,
            k: &mut self.k,
            kv_mem: &mut self.kv_mem,
            delta: &mut self.delta,
            out_head: &mut self.out_head,
        }
    }
}

/// Run both scans from the same state and rows; compare the output and the
/// advanced state bit for bit.
fn assert_scans_match(cfg: &Qwen35Config, seq_len: usize, warm: bool, seed: u64) {
    let mut rng = seed;
    let weights = weights(cfg, &mut rng);
    let mut start = GatedDeltaNetState::from_geometry(&require_qwen35_gdn_geometry(cfg));
    if warm {
        start.s_matrices = noise(&mut rng, start.s_matrices.len(), 0.5);
        start.conv_buffer = noise(&mut rng, start.conv_buffer.len(), 1.0);
    }
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();
    let qkv = noise(&mut rng, seq_len * qkv_dim, 1.0);
    let z = noise(&mut rng, seq_len * output_dim, 1.0);
    let beta: Vec<f32> = noise(&mut rng, seq_len * value_heads, 1.0)
        .into_iter()
        .map(|x| 0.5 + 0.5 * x)
        .collect();
    let alpha = noise(&mut rng, seq_len * value_heads, 2.0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let run = |by_head: bool| {
        let mut state = start.clone();
        let mut qkv = qkv.clone();
        let mut z = z.clone();
        let mut out = vec![f32::NAN; seq_len * output_dim];
        let mut workspace = Workspace::new(cfg);
        let rows = ScanRows {
            qkv: &mut qkv,
            z: &mut z,
            beta: &beta,
            alpha: &alpha,
            out: &mut out,
        };
        let work = workspace.work();
        pool.install(|| {
            if by_head {
                scan_by_head(cfg, &weights, &mut state, seq_len, rows, work);
            } else {
                scan_per_token(cfg, &weights, &mut state, seq_len, rows, work, 0);
            }
        });
        (if by_head { z } else { out }, state)
    };
    let (expected, expected_state) = run(false);
    let (actual, actual_state) = run(true);

    let case = format!(
        "heads={}/{} dims={}x{} seq={seq_len} warm={warm}",
        cfg.linear_num_key_heads, value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    );
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "{case}: reference output not finite"
    );
    assert_eq!(bits(&actual), bits(&expected), "{case}: output");
    assert_eq!(
        bits(&actual_state.s_matrices),
        bits(&expected_state.s_matrices),
        "{case}: recurrent state"
    );
    assert_eq!(
        bits(&actual_state.conv_buffer),
        bits(&expected_state.conv_buffer),
        "{case}: conv state"
    );
}

#[test]
fn heads_scan_matches_the_token_loop_bitwise_at_the_production_shape() {
    // 16 key and value heads of 128, the 0.8B checkpoint's shape; 197 rows make
    // four conv blocks, the last one short.
    let cfg = config(16, 16, 128, 128);
    for warm in [false, true] {
        assert_scans_match(&cfg, 3 * 64 + 5, warm, 0x9e37_79b9_7f4a_7c15);
    }
}

#[test]
fn heads_scan_matches_the_token_loop_bitwise_with_shared_key_heads() {
    // Two value heads per key head, as in the larger checkpoints.
    let cfg = config(2, 4, 32, 24);
    for warm in [false, true] {
        assert_scans_match(&cfg, 3 * 64 + 5, warm, 0x2545_f491_4f6c_dd1d);
    }
}

#[test]
fn heads_scan_matches_the_token_loop_bitwise_on_prompts_shorter_than_the_conv() {
    // Fewer rows than the conv reaches back: the next conv state keeps some of
    // the previous one.
    let cfg = config(2, 2, 16, 16);
    for seq_len in [1, 2, 3, 4, 5] {
        assert_scans_match(&cfg, seq_len, true, 0x51_7cc1_b727_220a + seq_len as u64);
    }
}

#[test]
fn heads_scan_matches_the_token_loop_bitwise_with_every_conv_block() {
    // 2113 rows make the most blocks the conv uses, 32 of 67 rows, the last one
    // 36 rows.
    let cfg = config(2, 2, 8, 8);
    assert_scans_match(&cfg, 2113, true, 0x94d0_49bb_1331_11eb);
}
