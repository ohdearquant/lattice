//! The facade's BERT-layer attention entry points against `lattice-compute`'s projection-view kernels.

use super::*;
use crate::attention::flash::{
    TiledAttentionBuffers, TiledAttentionConfig, tiled_multi_head_attention,
};
use crate::lora_hook::NoopLoraHook;
use crate::weights::{Tensor1D, Tensor2D};

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7919 + seed * 104_729) % 2003) as f32 / 2003.0 - 0.5) * 0.25)
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn matrix(data: &[f32], rows: usize, cols: usize) -> Tensor2D<'_> {
    Tensor2D { data, rows, cols }
}

fn vector(data: &[f32]) -> Tensor1D<'_> {
    Tensor1D {
        data,
        len: data.len(),
    }
}

/// One BERT layer whose eight attention tensors all hold different values, so a projection taken
/// from the wrong field changes the output.
struct LayerData {
    hidden: usize,
    kv_dim: usize,
    attention: [Vec<f32>; 8],
    unused: Vec<f32>,
}

impl LayerData {
    fn new(hidden: usize, kv_dim: usize) -> Self {
        let lens = [
            hidden * hidden,
            hidden,
            kv_dim * hidden,
            kv_dim,
            kv_dim * hidden,
            kv_dim,
            hidden * hidden,
            hidden,
        ];
        Self {
            hidden,
            kv_dim,
            attention: std::array::from_fn(|i| values(lens[i], i + 1)),
            unused: vec![0.0; hidden * hidden],
        }
    }

    fn layer(&self) -> TransformerLayerWeights<'_> {
        let (h, kv, a) = (self.hidden, self.kv_dim, &self.attention);
        let row = &self.unused[..h];
        TransformerLayerWeights {
            query_weight: matrix(&a[0], h, h),
            query_bias: vector(&a[1]),
            key_weight: matrix(&a[2], kv, h),
            key_bias: vector(&a[3]),
            value_weight: matrix(&a[4], kv, h),
            value_bias: vector(&a[5]),
            attn_output_weight: matrix(&a[6], h, h),
            attn_output_bias: vector(&a[7]),
            attn_layer_norm_weight: vector(row),
            attn_layer_norm_bias: vector(row),
            ffn_intermediate_weight: matrix(&self.unused, h, h),
            ffn_intermediate_bias: vector(row),
            ffn_output_weight: matrix(&self.unused, h, h),
            ffn_output_bias: vector(row),
            ffn_layer_norm_weight: vector(row),
            ffn_layer_norm_bias: vector(row),
        }
    }

    /// The view built field by field, independently of `bert_attention_projections`.
    fn projections(&self) -> AttentionProjections<'_> {
        let (h, kv, a) = (self.hidden, self.kv_dim, &self.attention);
        AttentionProjections {
            query_weight: matrix(&a[0], h, h),
            query_bias: vector(&a[1]),
            key_weight: matrix(&a[2], kv, h),
            key_bias: vector(&a[3]),
            value_weight: matrix(&a[4], kv, h),
            value_bias: vector(&a[5]),
            output_weight: matrix(&a[6], h, h),
            output_bias: vector(&a[7]),
        }
    }
}

#[test]
fn multi_head_attention_matches_the_projection_kernel_bit_for_bit() {
    let (seq_len, num_heads, head_dim) = (5, 2, 4);
    let hidden = num_heads * head_dim;
    let data = LayerData::new(hidden, hidden);
    let hidden_states = values(seq_len * hidden, 11);
    let mask = [1, 1, 0, 1, 1];
    let run = |projections: &AttentionProjections<'_>| {
        let mut buffers = AttentionBuffers::new(seq_len, hidden, num_heads, hidden);
        lattice_compute::attention::standard::multi_head_attention(
            &hidden_states,
            projections,
            &mask,
            seq_len,
            hidden,
            num_heads,
            head_dim,
            &mut buffers,
            &NoopLoraHook,
            0,
        )
    };

    let mut buffers = AttentionBuffers::new(seq_len, hidden, num_heads, hidden);
    let facade = multi_head_attention(
        &hidden_states,
        &data.layer(),
        &mask,
        seq_len,
        hidden,
        num_heads,
        head_dim,
        &mut buffers,
        &NoopLoraHook,
        0,
    );
    let kernel = run(&data.projections());
    assert!(facade.iter().any(|v| *v != 0.0));
    assert_eq!(bits(&facade), bits(&kernel));

    // Control: the same kernel with key and value exchanged gives a different output, so the
    // comparison above can see a projection read from the wrong field.
    let mut swapped = data.projections();
    std::mem::swap(&mut swapped.key_weight, &mut swapped.value_weight);
    std::mem::swap(&mut swapped.key_bias, &mut swapped.value_bias);
    assert_ne!(bits(&run(&swapped)), bits(&kernel));
}

#[test]
fn tiled_multi_head_attention_matches_the_projection_kernel_bit_for_bit() {
    let (seq_len, num_q_heads, num_kv_heads, head_dim) = (9, 4, 2, 4);
    let hidden = num_q_heads * head_dim;
    let data = LayerData::new(hidden, num_kv_heads * head_dim);
    let hidden_states = values(seq_len * hidden, 12);
    let mask = [1, 1, 1, 0, 1, 1, 1, 1, 0];
    let config = TiledAttentionConfig::new(num_q_heads, num_kv_heads, head_dim);
    let run = |projections: &AttentionProjections<'_>| {
        let mut buffers = AttentionBuffers::new(seq_len, hidden, num_q_heads, hidden);
        let mut tiled = TiledAttentionBuffers::new(seq_len, &config);
        lattice_compute::attention::flash::tiled_multi_head_attention(
            &hidden_states,
            projections,
            &mask,
            seq_len,
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut buffers,
            &mut tiled,
            &config,
        )
    };

    let mut buffers = AttentionBuffers::new(seq_len, hidden, num_q_heads, hidden);
    let mut tiled = TiledAttentionBuffers::new(seq_len, &config);
    let facade = tiled_multi_head_attention(
        &hidden_states,
        &data.layer(),
        &mask,
        seq_len,
        hidden,
        num_q_heads,
        num_kv_heads,
        head_dim,
        &mut buffers,
        &mut tiled,
        &config,
    );
    let kernel = run(&data.projections());
    assert!(facade.iter().any(|v| *v != 0.0));
    assert_eq!(bits(&facade), bits(&kernel));

    let mut swapped = data.projections();
    std::mem::swap(&mut swapped.key_weight, &mut swapped.value_weight);
    std::mem::swap(&mut swapped.key_bias, &mut swapped.value_bias);
    assert_ne!(bits(&run(&swapped)), bits(&kernel));
}
