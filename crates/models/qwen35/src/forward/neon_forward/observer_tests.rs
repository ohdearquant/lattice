use super::*;
use crate::gdn_state_observer::{GdnStateObservation, with_observer};
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn q8_neon_observer_reports_mixed_layer_and_positions_without_mutation() {
    let (mut cfg, mut weights, rope) = make_nonzero_q8_neon_test_model();
    cfg.layer_types.swap(0, 1);
    weights.layers.swap(0, 1);
    let geometry = require_qwen35_gdn_geometry(&cfg);
    let mut expected = vec![GatedDeltaNetState::from_geometry(&geometry)];
    let mut actual = vec![GatedDeltaNetState::from_geometry(&geometry)];
    let mut expected_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut actual_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut expected_scratch = ForwardScratch::new();
    let mut actual_scratch = ForwardScratch::new();
    let events = Rc::new(RefCell::new(Vec::new()));
    for (position, token) in [7, 11].into_iter().enumerate() {
        forward_step_q8_neon(
            &weights,
            &cfg,
            &rope,
            token,
            position,
            &mut expected,
            &mut expected_cache,
            &mut expected_scratch,
        );
        let sink = Rc::clone(&events);
        with_observer(
            move |event| sink.borrow_mut().push(event),
            || {
                forward_step_q8_neon(
                    &weights,
                    &cfg,
                    &rope,
                    token,
                    position,
                    &mut actual,
                    &mut actual_cache,
                    &mut actual_scratch,
                );
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
    for (position, event) in observed.iter().enumerate() {
        let GdnStateObservation::Health {
            layer,
            position: actual_position,
            total,
            heads,
        } = event
        else {
            panic!("health event required");
        };
        assert_eq!((*layer, *actual_position), (1, position));
        assert_eq!((total.nan_count, total.infinite_count), (0, 0));
        assert_eq!(heads.len(), cfg.linear_num_value_heads());
    }
    let total = actual[0]
        .s_matrices
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(total > 0.0, "fixture must exercise nonzero recurrent state");
    let GdnStateObservation::Health {
        total: health,
        heads,
        ..
    } = &observed[1]
    else {
        panic!("health event required");
    };
    assert!((health.frobenius_norm.unwrap() - total).abs() < 1e-12);
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
