use super::*;

#[test]
fn test_full_workflow() {
    let mut network = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .hidden(4, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build()
        .unwrap();

    assert_eq!(network.num_inputs(), 4);
    assert_eq!(network.num_outputs(), 2);
    assert_eq!(network.num_layers(), 3);

    let input = [1.0, 2.0, 3.0, 4.0];
    let output = network.forward(&input).unwrap();

    // Softmax output sums to 1
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_inference_speed() {
    use std::time::Instant;

    // Create a reasonably sized network
    let mut network = NetworkBuilder::new()
        .input(128)
        .hidden(256, Activation::ReLU)
        .hidden(128, Activation::ReLU)
        .hidden(64, Activation::ReLU)
        .output(10, Activation::Softmax)
        .build()
        .unwrap();

    let input: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
    let output = network.forward(&input).unwrap();
    assert_eq!(output.len(), 10);
    assert!(output.iter().all(|value| value.is_finite()));

    for _ in 0..10 {
        network.forward(&input).unwrap();
    }

    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        network.forward(&input).unwrap();
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    println!("Average inference time: {avg_us:.2} us");

    // Wall-clock timing flakes under shared-CI load, so assert the bound only
    // on a quiet non-CI machine (CI runners set the CI env var). The forward
    // passes above still run in CI as a smoke check that inference does not
    // panic or hang; perf regressions are tracked by the Criterion benches.
    if std::env::var_os("CI").is_none() {
        assert!(avg_us < 5000.0, "Inference too slow: {avg_us} us");
    }
}

#[test]
fn test_deterministic_output() {
    let mut network = NetworkBuilder::new()
        .input(4)
        .output(2, Activation::Linear)
        .build()
        .unwrap();

    let input = [1.0, 2.0, 3.0, 4.0];

    let output1 = network.forward(&input).unwrap().to_vec();
    let output2 = network.forward(&input).unwrap().to_vec();

    // Should be deterministic
    for (a, b) in output1.iter().zip(output2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_parameter_count() {
    let network = NetworkBuilder::new()
        .input(784) // MNIST input
        .hidden(128, Activation::ReLU) // 784*128 + 128 = 100480
        .hidden(64, Activation::ReLU) // 128*64 + 64 = 8256
        .output(10, Activation::Softmax) // 64*10 + 10 = 650
        .build()
        .unwrap();

    assert_eq!(network.total_params(), 100480 + 8256 + 650);
}
