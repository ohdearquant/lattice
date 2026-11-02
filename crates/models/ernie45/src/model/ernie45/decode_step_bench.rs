//! Manual timing of one cached decode step at the PaddleOCR-VL decoder's
//! dimensions, with deterministic synthetic weights.
//!
//! Weight construction and the prompt prefill are outside the timed region.
//! Each sample times one call of the decode step the cached request loop
//! makes, so prefill and vision cost cannot hide a per-step change. The logits
//! checksum is printed so two builds can be checked for identical arithmetic.

use super::tests::{equiv_weights, pseudo_random_vec};
use super::*;
use std::hint::black_box;
use std::time::Instant;

const PREFIX: usize = 256;
const STEPS: usize = 96;
const BLOCK: usize = 32;

fn decoder_config() -> Ernie45Config {
    Ernie45Config {
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 18,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: 128,
        vocab_size: 103424,
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        rope_scaling: Ernie45RopeScaling {
            mrope_section: vec![16, 24, 24],
        },
        tie_word_embeddings: false,
        use_bias: false,
    }
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

#[test]
#[ignore = "manual release timing; run under benchmark admission"]
fn bench_cached_decode_step() {
    if cfg!(debug_assertions) {
        panic!("run the decode step timing with --release");
    }
    let cfg = decoder_config();
    let hidden = cfg.hidden_size;
    let model = Ernie45Model::new(cfg.clone(), equiv_weights(&cfg)).expect("synthetic decoder");
    let capacity = PREFIX + STEPS;
    let mut cache = Ernie45KvCache::new(
        cfg.num_hidden_layers,
        cfg.kv_dim().expect("kv dim"),
        capacity,
    )
    .expect("decode cache");
    let positions: Vec<[u32; 3]> = (0..PREFIX as u32).map(|p| [p, p, p]).collect();
    let prefill_start = Instant::now();
    model
        .kv_prefill(
            &pseudo_random_vec(PREFIX * hidden, 0x5eed_0001),
            &positions,
            &mut cache,
        )
        .expect("prefill");
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;
    let step_embeds = pseudo_random_vec(STEPS * hidden, 0x5eed_0002);
    let mut scratch = Ernie45DecodeScratch::new(&cfg, capacity).expect("decode scratch");

    let mut step_ns = Vec::with_capacity(STEPS);
    let mut checksum = 0.0f64;
    for step in 0..STEPS {
        let p = (PREFIX + step) as u32;
        let embed = &step_embeds[step * hidden..][..hidden];
        let start = Instant::now();
        let logits = model
            .kv_decode_step_with_scratch(black_box(embed), [p, p, p], &mut cache, &mut scratch)
            .expect("decode step");
        let elapsed = start.elapsed().as_nanos();
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
        checksum += logits.iter().map(|&value| f64::from(value)).sum::<f64>();
        black_box(logits);
        step_ns.push(elapsed);
    }

    for (block, chunk) in step_ns.chunks(BLOCK).enumerate() {
        let mut chunk = chunk.to_vec();
        let min = chunk.iter().copied().min().unwrap_or_default();
        eprintln!(
            "ernie_decode_step block={block} steps={} median_ns={} min_ns={min}",
            chunk.len(),
            median(&mut chunk),
        );
    }
    let mut all = step_ns.clone();
    eprintln!(
        "ernie_decode_step prefix={PREFIX} steps={STEPS} prefill_ms={prefill_ms:.1} median_ns={} min_ns={} checksum={checksum:.6}",
        median(&mut all),
        step_ns.iter().copied().min().unwrap_or_default(),
    );
}
