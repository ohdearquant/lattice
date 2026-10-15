use super::{TiledAttentionBuffers, TiledAttentionConfig, causal_tiled_attention_head};
use std::hint::black_box;
use std::time::Instant;

fn fixture(len: usize, mut seed: u64) -> Vec<f32> {
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.5
        })
        .collect()
}

#[test]
#[ignore = "manual CPU timing; run under benchmark admission"]
fn bench_causal_prefill_attention() {
    const ITERATIONS: usize = 5;
    for head_dim in [32, 128] {
        for seq_len in [64, 257, 1024] {
            let q_stride = 4 * head_dim;
            let kv_stride = 2 * head_dim;
            let out_stride = 2 * head_dim;
            let q_offset = head_dim;
            let kv_offset = head_dim;
            let q = fixture(seq_len * q_stride, 0x9182_7354);
            let k = fixture(seq_len * kv_stride, 0x2536_4918);
            let v = fixture(seq_len * kv_stride, 0x8321_5946);
            let mut output = vec![0.0; seq_len * out_stride];
            let config = TiledAttentionConfig::new(2, 1, head_dim);
            let mut buffers = TiledAttentionBuffers::new(seq_len, &config);
            let mut acc = vec![0.0; head_dim];
            let mut scores = vec![0.0; config.tile_size_kv.max(1)];
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut run = || {
                causal_tiled_attention_head(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    &mut output,
                    seq_len,
                    q_stride,
                    kv_stride,
                    out_stride,
                    q_offset,
                    kv_offset,
                    head_dim,
                    scale,
                    &config,
                    &mut buffers,
                    &mut acc,
                    &mut scores,
                );
                black_box(&output);
            };
            run();
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                run();
            }
            let elapsed = start.elapsed();
            let checksum: f64 = output.iter().map(|&value| f64::from(value)).sum();
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.iter().any(|value| value.abs() > 1e-6));
            eprintln!(
                "causal_prefill_head seq={seq_len} dim={head_dim} iterations={ITERATIONS} elapsed_ns={} ns_per_call={:.0} checksum={checksum:.9}",
                elapsed.as_nanos(),
                elapsed.as_nanos() as f64 / ITERATIONS as f64,
            );
        }
    }
}
