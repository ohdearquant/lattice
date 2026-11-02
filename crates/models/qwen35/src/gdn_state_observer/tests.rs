use super::*;
use crate::attention::gdn::require_qwen35_gdn_geometry;
use crate::model::qwen35::{ForwardScratch, KvCache, test_support::tiny_zero_model};
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn summaries_count_nonfinite_per_head_and_refuse_malformed_shapes() {
    let values = [3.0, 4.0, 0.0, -0.0, f32::NAN, f32::INFINITY, 1.0, -1.0];
    let GdnStateObservation::Health {
        layer,
        position,
        total,
        heads,
    } = summarize(7, 11, &values, 2, 2, 2)
    else {
        panic!("valid layout")
    };
    assert_eq!((layer, position), (7, 11));
    assert_eq!(
        (total.finite_count, total.nan_count, total.infinite_count),
        (6, 1, 1)
    );
    assert_eq!(total.frobenius_norm, None);
    assert_eq!(heads[0].frobenius_norm, Some(5.0));
    assert_eq!(
        (
            heads[1].finite_count,
            heads[1].nan_count,
            heads[1].infinite_count
        ),
        (2, 1, 1)
    );
    assert_eq!(heads[1].frobenius_norm, None);
    let nonfinite = health(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(
        (
            nonfinite.finite_count,
            nonfinite.nan_count,
            nonfinite.infinite_count
        ),
        (0, 1, 2)
    );
    assert_eq!(nonfinite.frobenius_norm, None);
    let GdnStateObservation::Health { total, .. } = summarize(0, 0, &[f32::MAX], 1, 1, 1) else {
        panic!("valid extreme")
    };
    assert_eq!(total.frobenius_norm, Some(f64::from(f32::MAX)));
    assert!(matches!(
        summarize(3, 9, &[1.0], 2, 2, 2),
        GdnStateObservation::InvalidLayout {
            expected: Some(8),
            actual: 1,
            ..
        }
    ));
    assert!(matches!(
        summarize(3, 9, &[], usize::MAX, usize::MAX, 2),
        GdnStateObservation::InvalidLayout {
            expected: None,
            actual: 0,
            ..
        }
    ));
    assert!(matches!(
        summarize(3, 9, &[], 0, 1, 1),
        GdnStateObservation::InvalidLayout { expected: None, .. }
    ));
}

#[test]
fn scopes_restore_the_previous_observer_even_on_unwind() {
    let model = tiny_zero_model();
    let state = GatedDeltaNetState::from_geometry(&require_qwen35_gdn_geometry(&model.config));
    let events = Rc::new(RefCell::new(Vec::new()));
    let outer = Rc::clone(&events);
    with_observer(
        move |_| outer.borrow_mut().push("outer"),
        || {
            observe(0, 0, &state, &model.config);
            let inner = Rc::clone(&events);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_observer(
                    move |_| inner.borrow_mut().push("inner"),
                    || {
                        observe(0, 1, &state, &model.config);
                        panic!("exercise scope restoration");
                    },
                );
            }));
            assert!(result.is_err());
            observe(0, 2, &state, &model.config);
        },
    );
    observe(0, 3, &state, &model.config);
    assert_eq!(*events.borrow(), ["outer", "inner", "outer"]);
}

#[test]
fn f32_cpu_steps_emit_per_layer_without_changing_state_or_logits() {
    let model = tiny_zero_model();
    let geometry = require_qwen35_gdn_geometry(&model.config);
    let build = || {
        (0..model.config.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::from_geometry(&geometry))
            .collect::<Vec<_>>()
    };
    let mut reference = build();
    let mut observed = build();
    let mut reference_cache = KvCache::new(model.config.num_full_attention_layers());
    let mut observed_cache = KvCache::new(model.config.num_full_attention_layers());
    let mut reference_scratch = ForwardScratch::new();
    let mut observed_scratch = ForwardScratch::new();
    model.forward_step(
        0,
        0,
        &mut reference,
        &mut reference_cache,
        &mut reference_scratch,
    );
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&events);
    with_observer(
        move |event| sink.borrow_mut().push(event),
        || {
            model.forward_step(
                0,
                0,
                &mut observed,
                &mut observed_cache,
                &mut observed_scratch,
            );
        },
    );
    assert_eq!(observed_scratch.logits, reference_scratch.logits);
    assert_eq!(
        observed
            .iter()
            .map(GatedDeltaNetState::snapshot)
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(GatedDeltaNetState::snapshot)
            .collect::<Vec<_>>()
    );
    let events = events.borrow();
    assert_eq!(events.len(), model.config.num_linear_attention_layers());
    for (layer, event) in events.iter().enumerate() {
        let GdnStateObservation::Health {
            layer: actual,
            position,
            total,
            heads,
        } = event
        else {
            panic!("healthy state expected")
        };
        assert_eq!((*actual, *position), (layer, 0));
        assert_eq!(total.frobenius_norm, Some(0.0));
        assert_eq!((total.nan_count, total.infinite_count), (0, 0));
        assert_eq!(heads.len(), model.config.linear_num_value_heads());
    }
}

#[test]
fn batched_prefill_emits_each_layer_and_position_without_changing_outputs() {
    let model = tiny_zero_model();
    let geometry = require_qwen35_gdn_geometry(&model.config);
    let build = || {
        (0..model.config.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::from_geometry(&geometry))
            .collect::<Vec<_>>()
    };
    let mut reference = build();
    let mut observed = build();
    let mut reference_cache = KvCache::new(model.config.num_full_attention_layers());
    let mut observed_cache = KvCache::new(model.config.num_full_attention_layers());
    let expected = model
        .prefill_tokens_batched_for_generate(&[0, 1, 2], &mut reference, &mut reference_cache)
        .unwrap();
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&events);
    let actual = with_observer(
        move |event| sink.borrow_mut().push(event),
        || {
            model
                .prefill_tokens_batched_for_generate(&[0, 1, 2], &mut observed, &mut observed_cache)
                .unwrap()
        },
    );
    assert_eq!(actual, expected);
    assert_eq!(
        observed
            .iter()
            .map(GatedDeltaNetState::snapshot)
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(GatedDeltaNetState::snapshot)
            .collect::<Vec<_>>()
    );
    let events = events.borrow();
    assert_eq!(events.len(), model.config.num_linear_attention_layers() * 3);
    for (i, event) in events.iter().enumerate() {
        let GdnStateObservation::Health {
            layer,
            position,
            total,
            ..
        } = event
        else {
            panic!("healthy state expected")
        };
        assert_eq!((*layer, *position), (i / 3, i % 3));
        assert_eq!(total.frobenius_norm, Some(0.0));
    }
}

#[test]
fn single_token_debug_reports_mixed_layer_health_without_changing_logits() {
    use crate::model::qwen35::AttentionWeights;

    let mut model = tiny_zero_model();
    // Move the full-attention layer before the GDN layers, so model indices
    // differ from the compact recurrent-state indices.
    model.config.layer_types.swap(0, 3);
    model.weights.layers.swap(0, 3);
    let hidden = model.config.hidden_size;
    model.weights.embed_tokens[hidden..2 * hidden].fill(0.05);
    for (attention, _) in &mut model.weights.layers {
        if let AttentionWeights::Linear(weights) = attention {
            weights.in_proj_qkv.fill(0.002);
            weights.conv1d_weight.fill(0.01);
        }
    }
    let baseline = model.forward_single_token_debug(1);
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&events);
    let observed_logits = with_observer(
        move |event| sink.borrow_mut().push(event),
        || model.forward_single_token_debug(1),
    );
    assert_eq!(baseline, observed_logits);

    let geometry = require_qwen35_gdn_geometry(&model.config);
    let mut states: Vec<_> = (0..model.config.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::from_geometry(&geometry))
        .collect();
    let mut cache = KvCache::new(model.config.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    let reference_events = Rc::new(RefCell::new(Vec::new()));
    let reference_sink = Rc::clone(&reference_events);
    with_observer(
        move |event| reference_sink.borrow_mut().push(event),
        || model.forward_step(1, 0, &mut states, &mut cache, &mut scratch),
    );
    assert_eq!(observed_logits, scratch.logits);
    let events = events.borrow();
    let reference_events = reference_events.borrow();
    assert_eq!(*events, *reference_events);
    assert_eq!(events.len(), 3);
    for (idx, event) in events.iter().enumerate() {
        let GdnStateObservation::Health {
            layer,
            position,
            total,
            heads,
        } = event
        else {
            panic!("health event required");
        };
        assert_eq!((*layer, *position), (idx + 1, 0));
        assert_eq!((total.nan_count, total.infinite_count), (0, 0));
        assert!(
            total.frobenius_norm.unwrap() > 0.0,
            "fixture must update state"
        );
        assert_eq!(heads.len(), model.config.linear_num_value_heads());
    }
}
