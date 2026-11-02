use crate::model::qwen35::{ForwardScratch, KvCache};
use lattice_arch::qwen35_config::{LayerType, Qwen35Config};

pub(crate) const SENTINEL: f32 = 147.5;

pub(crate) fn config() -> Qwen35Config {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.hidden_size = 64;
    cfg.num_hidden_layers = 2;
    cfg.vocab_size = 128;
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 32;
    cfg.linear_num_key_heads = 2;
    cfg.linear_num_value_heads = Some(2);
    cfg.linear_key_head_dim = 32;
    cfg.linear_value_head_dim = 32;
    cfg.full_attention_interval = 2;
    cfg.layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];
    cfg.layer_mask = vec![true; 2];
    cfg.eos_token_id = 127;
    cfg.max_position_embeddings = 512;
    cfg.mtp_num_hidden_layers = 0;
    cfg
}

pub(crate) fn projections(cfg: &Qwen35Config) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden_size;
    let mut q = vec![0.0; 2 * cfg.full_q_dim() * hidden];
    for h in 0..cfg.num_attention_heads {
        for d in 0..cfg.head_dim {
            q[(2 * h * cfg.head_dim + d) * hidden + (d + 11 * h) % hidden] = 1.0;
        }
    }
    let mut k = vec![0.0; cfg.full_kv_dim() * hidden];
    let mut v = k.clone();
    for d in 0..cfg.full_kv_dim() {
        k[d * hidden + d % hidden] = 1.0;
        v[d * hidden + (d + 7) % hidden] = 1.0;
    }
    (q, k, v)
}

pub(crate) fn cache(cfg: &Qwen35Config, seq_len: usize) -> KvCache {
    let mut cache = KvCache::new(2);
    let kv_dim = cfg.full_kv_dim();
    cache.reserve(300, kv_dim);
    cache.k[0] = vec![f32::NAN; (seq_len + 1) * kv_dim];
    cache.v[0] = cache.k[0].clone();
    for t in 0..seq_len {
        for h in 0..cfg.num_key_value_heads {
            for d in 0..cfg.head_dim {
                cache.k[1].push((t + 1) as f32 * (1 + h + d % 3) as f32 * 0.01);
                cache.v[1].push((t * 7 + h * 31 + d) as f32 * 0.125 - 4.0);
            }
        }
    }
    cache.seq_len = seq_len;
    cache
}

pub(crate) fn reset_scratch(scratch: &mut ForwardScratch, cfg: &Qwen35Config) {
    scratch.ensure_capacity(cfg, 300);
    scratch.context.resize(cfg.full_q_dim() + 7, SENTINEL);
    scratch.context.fill(SENTINEL);
    scratch.scores.fill(f32::NAN);
    for (i, x) in scratch.attn_out[..cfg.hidden_size].iter_mut().enumerate() {
        *x = (i as f32 + 1.0) * 0.03125;
    }
}

pub(crate) fn assert_context(
    context: &[f32],
    scores: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    gates: Option<&[f32]>,
) {
    let kv_dim = kv_heads * head_dim;
    assert_eq!(context.len(), heads * head_dim + 7);
    assert!(scores.len() > heads * seq_len);
    for h in 0..heads {
        let row = &scores[h * seq_len..(h + 1) * seq_len];
        assert!(row.iter().all(|x| x.is_finite()));
        if seq_len > 1 {
            assert!(row.windows(2).any(|w| w[0].to_bits() != w[1].to_bits()));
        }
        for d in 0..head_dim {
            // Keep the pre-change dimension-outer reduction as the independent oracle.
            let mut expected = 0.0f32;
            for t in 0..seq_len {
                expected += scores[h * seq_len + t]
                    * values[t * kv_dim + (h / (heads / kv_heads)) * head_dim + d];
            }
            let i = h * head_dim + d;
            if let Some(gates) = gates {
                expected *= 1.0 / (1.0 + (-gates[i]).exp());
            }
            assert!(expected.is_finite());
            assert_eq!(
                context[i].to_bits(),
                expected.to_bits(),
                "head {h}, lane {d}"
            );
        }
    }
    if seq_len > 1 && heads > 1 {
        assert!(
            scores[..seq_len]
                .iter()
                .zip(&scores[seq_len..2 * seq_len])
                .any(|(a, b)| a.to_bits() != b.to_bits())
        );
    }
    assert!(
        context[heads * head_dim..]
            .iter()
            .all(|x| x.to_bits() == SENTINEL.to_bits())
    );
    assert!(scores[heads * seq_len..].iter().all(|x| x.is_nan()));
}
