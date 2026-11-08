use super::*;

#[test]
fn packed_attention_capacities_match_the_existing_working_set_estimate() {
    for (hidden, heads, intermediate, cu_seqlens, longest, total) in [
        (12, 3, 29, [0, 7, 7, 10, 17, 19], 7, 19),
        (48, 6, 80, [0, 17, 19, 19, 36, 53], 17, 53),
        (384, 12, 1536, [0, 128, 128, 129, 256, 384], 128, 384),
    ] {
        let mut config = BertConfig::bge_small();
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.intermediate_size = intermediate;
        let head_dim = config.head_dim();
        let scratch = PackedAttentionScratch::new(&cu_seqlens, hidden, head_dim);
        let buffers = [
            &scratch.q_head,
            &scratch.k_head,
            &scratch.v_all_t,
            &scratch.scores_head,
            &scratch.context_head,
        ];
        let lengths = [
            longest * head_dim,
            longest * head_dim,
            longest * hidden,
            longest * longest,
            longest * head_dim,
        ];
        for (buffer, expected) in buffers.into_iter().zip(lengths) {
            assert_eq!(buffer.len(), expected);
            assert_eq!(buffer.capacity(), expected);
        }
        let actual_scratch_floats: usize = buffers.iter().map(|buffer| buffer.capacity()).sum();
        let existing_forward_floats = 10 * total * hidden + total * intermediate;
        let actual_bytes =
            ((existing_forward_floats + actual_scratch_floats) * std::mem::size_of::<f32>()) as u64;
        assert_eq!(
            packed_encode_working_set_bytes(Some(total), longest, &config),
            Some(actual_bytes)
        );
    }
}
