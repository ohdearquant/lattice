use lattice_fann::{Activation, Network, NetworkBuilder};
use lattice_inference::mixture::{AdapterRouter, RouterError};
use lattice_tune::lora::router_update::{
    DiagonalFisher, FeedbackEvent, PreferenceSignal, ReplayBuffer, RouterUpdateConfig,
    update_router,
};

fn gate(inputs: usize, outputs: usize) -> Network {
    NetworkBuilder::new()
        .input(inputs)
        .hidden(16, Activation::Tanh)
        .output(outputs, Activation::Linear)
        .build_with_seed(42)
        .unwrap()
}

#[test]
fn feedback_refit_reload_changes_selection() {
    let network = gate(2, 2);
    let bytes = network.to_bytes();
    let mut router = AdapterRouter::new(network);
    let context = vec![1.0, 0.5];
    let pool = vec!["first".to_string(), "second".to_string()];
    let before = router.route(&context, &pool, 1).unwrap();
    let preferred = 1 - usize::from(before[0].0 == pool[1]);
    let events = vec![
        FeedbackEvent {
            context_vector: context.clone(),
            preferred_adapter_idx: preferred,
            adapter_id: pool[preferred].clone(),
            signal: PreferenceSignal::Positive,
        };
        16
    ];
    let config = RouterUpdateConfig {
        learning_rate: 0.1,
        epochs: 40,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
        replay_mix_fraction: 0.0,
        ..RouterUpdateConfig::default()
    };
    let delta = update_router(
        &bytes,
        &events,
        &mut ReplayBuffer::new(32),
        &mut DiagonalFisher::new(0, 0.99).unwrap(),
        &config,
    )
    .unwrap();
    assert_eq!(delta.events_consumed, events.len());
    assert_eq!(router.route(&context, &pool, 1).unwrap(), before);
    router.reload(&delta.network_bytes).unwrap();
    let after = router.route(&context, &pool, 1).unwrap();
    assert_ne!(before, after, "feedback reload must change selection");
    assert_eq!(after, vec![(pool[preferred].clone(), 1.0)]);
}

#[test]
fn rejected_reload_preserves_original_route() {
    for (inputs, outputs) in [(3, 2), (2, 3), (3, 3)] {
        let mut router = AdapterRouter::new(gate(2, 2));
        let context = [1.0, 0.5];
        let pool = vec!["first".into(), "second".into()];
        let before = router.route(&context, &pool, 1).unwrap();
        let error = router
            .reload(&gate(inputs, outputs).to_bytes())
            .unwrap_err();
        assert!(matches!(
            error,
            RouterError::GateDimensionMismatch {
                expected_inputs: 2,
                expected_outputs: 2,
                got_inputs,
                got_outputs,
            } if got_inputs == inputs && got_outputs == outputs
        ));
        assert_eq!(router.route(&context, &pool, 1).ok(), Some(before));
        assert_eq!(router.input_size(), 2);
        assert_eq!(router.output_size(), 2);
    }
}
