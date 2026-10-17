use super::*;
use crate::activation::Activation;

fn sample_network() -> Network {
    // 3 -> 4 (ReLU) -> 2 (Linear): first layer has 3*4 = 12 weights.
    NetworkBuilder::new()
        .input(3)
        .hidden(4, Activation::ReLU)
        .output(2, Activation::Linear)
        .build()
        .unwrap()
}

#[test]
fn roundtrip_rebuilds_skipped_buffers_and_runs_forward() {
    let net = sample_network();
    let json = serde_json::to_string(&net).unwrap();
    // `buffers` is #[serde(skip)] — a direct field-deserialize would leave it
    // empty and panic in forward. The TryFrom path must rebuild it.
    let mut restored: Network = serde_json::from_str(&json).unwrap();
    let out = restored.forward(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(out.len(), 2);
}

#[test]
fn empty_layer_stack_is_rejected() {
    let result = serde_json::from_str::<Network>(r#"{"layers":[]}"#);
    assert!(
        result.is_err(),
        "empty layer stack must be rejected, got {result:?}"
    );
}

#[test]
fn layer_with_short_weights_is_rejected() {
    // Serialize a valid network, then drop one weight from the first layer
    // so weights.len() no longer equals num_inputs * num_outputs. Routing
    // through Value avoids hard-coding the Activation serialization format.
    let net = sample_network();
    let json = serde_json::to_string(&net).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let weights = value["layers"][0]["weights"].as_array_mut().unwrap();
    weights.pop();
    let corrupted = serde_json::to_string(&value).unwrap();

    let result = serde_json::from_str::<Network>(&corrupted);
    assert!(
        result.is_err(),
        "layer with short weights must be rejected, got {result:?}"
    );
}
