use super::*;
use crate::gdn_state_observer::{GdnStateObservation, with_observer};
use std::cell::RefCell;
use std::rc::Rc;

fn mixed_model() -> (Qwen35Config, F16ModelWeights, RopeTable) {
    let (mut cfg, mut weights, rope) = make_nonzero_f16_cpu_test_model();
    cfg.layer_types.swap(0, 1);
    weights.layers.swap(0, 1);
    (cfg, weights, rope)
}

#[test]
fn f16_observer_reports_real_layer_and_positions_without_mutation() {
    let (cfg, weights, rope) = mixed_model();
    let geometry = require_qwen35_gdn_geometry(&cfg);
    let mut expected = vec![GatedDeltaNetState::from_geometry(&geometry)];
    let mut actual = vec![GatedDeltaNetState::from_geometry(&geometry)];
    let mut expected_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut actual_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut expected_scratch = ForwardScratch::new();
    let mut actual_scratch = ForwardScratch::new();
    let events = Rc::new(RefCell::new(Vec::new()));
    for (position, token) in [1, 2].into_iter().enumerate() {
        forward_step_f16(
            &weights,
            &cfg,
            &rope,
            token,
            position,
            &mut expected,
            &mut expected_cache,
            &mut expected_scratch,
            None,
            None,
        )
        .unwrap();
        let sink = Rc::clone(&events);
        with_observer(
            move |event| sink.borrow_mut().push(event),
            || {
                forward_step_f16(
                    &weights,
                    &cfg,
                    &rope,
                    token,
                    position,
                    &mut actual,
                    &mut actual_cache,
                    &mut actual_scratch,
                    None,
                    None,
                )
                .unwrap();
            },
        );
        assert_eq!(expected_scratch.logits, actual_scratch.logits);
        assert_eq!(expected[0].snapshot(), actual[0].snapshot());
        assert_eq!(expected_cache.k, actual_cache.k);
        assert_eq!(expected_cache.v, actual_cache.v);
        expected_cache.seq_len += 1;
        actual_cache.seq_len += 1;
    }
    let observed = events.borrow();
    assert_eq!(observed.len(), 2);
    for (i, event) in observed.iter().enumerate() {
        let GdnStateObservation::Health {
            layer,
            position,
            total,
            heads,
        } = event
        else {
            panic!("health event required");
        };
        assert_eq!((*layer, *position), (1, i));
        assert_eq!(total.nan_count, 0);
        assert_eq!(total.infinite_count, 0);
        assert_eq!(heads.len(), cfg.linear_num_value_heads());
    }
    let total = actual[0]
        .s_matrices
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(total > 0.0, "fixture must exercise nonzero recurrent state");
    let GdnStateObservation::Health { total: last, .. } = &observed[1] else {
        panic!("health event required");
    };
    assert!((last.frobenius_norm.unwrap() - total).abs() < 1e-12);
    let GdnStateObservation::Health { heads, .. } = &observed[1] else {
        panic!("health event required");
    };
    let head_size = cfg.linear_key_head_dim * cfg.linear_value_head_dim;
    for (head, values) in heads
        .iter()
        .zip(actual[0].s_matrices.chunks_exact(head_size))
    {
        let norm = values
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        assert_eq!(head.finite_count, head_size);
        assert_eq!((head.nan_count, head.infinite_count), (0, 0));
        assert!((head.frobenius_norm.unwrap() - norm).abs() < 1e-12);
    }
}

#[test]
fn multimodal_image_f16_path_emits_observations() {
    use crate::vision::multimodal::Qwen35VisionRequest;
    use crate::vision::qwen35_vit::GridThw;
    use lattice_arch::qwen35_config::{RopeParams, VisionModelConfig};

    let (mut cfg, weights, _) = mixed_model();
    cfg.image_token_id = Some(3);
    cfg.rope_parameters = Some(RopeParams {
        rope_theta: cfg.rope_theta,
        partial_rotary_factor: Some(cfg.partial_rotary_factor),
        mrope_section: Some(vec![3, 3, 2]),
        mrope_interleaved: Some(true),
    });
    cfg.vision_config = Some(VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 1,
        patch_size: 1,
        spatial_merge_size: 2,
        out_hidden_size: cfg.hidden_size,
        temporal_patch_size: 1,
        num_position_embeddings: 1,
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    });
    let request = Qwen35VisionRequest {
        input_ids: vec![0, 3, 3, 3, 3, 1],
        image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
        post_merger_rows: (0..4 * cfg.hidden_size)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect(),
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: cfg.hidden_size,
    };
    let mut gen_cfg = GenerateConfig::default();
    gen_cfg.max_new_tokens = 2;
    gen_cfg.temperature = 0.0;
    gen_cfg.seed = Some(1);
    gen_cfg.stop_token_ids = vec![];
    let baseline = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).unwrap();
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&events);
    let output = with_observer(
        move |event| sink.borrow_mut().push(event),
        || generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).unwrap(),
    );
    assert_eq!(output.token_ids, baseline.token_ids);
    assert_eq!(output.generated_tokens, baseline.generated_tokens);
    assert_eq!(output.stop_reason, baseline.stop_reason);
    let events = events.borrow();
    assert_eq!(
        events.len(),
        request.input_ids.len() + 1,
        "six prefill and one decode update"
    );
    for (position, event) in events.iter().enumerate() {
        let GdnStateObservation::Health {
            layer,
            position: actual,
            ..
        } = event
        else {
            panic!("health event required");
        };
        assert_eq!((*layer, *actual), (1, position));
    }
}
