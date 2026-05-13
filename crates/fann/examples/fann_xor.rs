//! Train a tiny XOR network with lattice-fann.
//!
//! Run with:
//!   cargo run -p lattice-fann --example fann_xor
//!
//! XOR is the canonical "cannot be solved by a linear classifier" problem.
//! A 2->4->1 network with Tanh activations learns it reliably in ~500-2000
//! epochs depending on random initialization and learning-rate settings.
//!
//! This example also demonstrates:
//! - NetworkBuilder fluent API
//! - Seeded initialization for reproducibility
//! - TrainingConfig builder methods
//! - TrainingResult inspection
//! - Intermediate activation access

use lattice_fann::{Activation, BackpropTrainer, NetworkBuilder, Trainer, TrainingConfig};

fn main() {
    // -------------------------------------------------------------------------
    // 1. Build the network
    // -------------------------------------------------------------------------
    // Tanh outputs in [-1, 1], which matches our XOR targets below.
    // Using build_with_seed for a reproducible starting point.
    let mut network = NetworkBuilder::new()
        .input(2)
        .hidden(4, Activation::Tanh)
        .output(1, Activation::Tanh)
        .build_with_seed(42)
        .expect("valid architecture");

    println!("Architecture: {}", network.architecture());
    println!("Total params: {}", network.total_params()); // 2*4+4 + 4*1+1 = 17
    println!();

    // -------------------------------------------------------------------------
    // 2. XOR truth table
    // -------------------------------------------------------------------------
    // Tanh saturates toward +/-1; mapping true->+1, false->-1 works well
    // with Tanh activations and MSE loss.
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0], // XOR = 0 => target -1
        vec![0.0, 1.0], // XOR = 1 => target +1
        vec![1.0, 0.0], // XOR = 1 => target +1
        vec![1.0, 1.0], // XOR = 0 => target -1
    ];
    let targets: Vec<Vec<f32>> = vec![vec![-1.0], vec![1.0], vec![1.0], vec![-1.0]];

    // -------------------------------------------------------------------------
    // 3. Training configuration
    // -------------------------------------------------------------------------
    let config = TrainingConfig::new()
        .learning_rate(0.5)
        .momentum(0.9)
        .weight_decay(0.0) // no regularization for tiny dataset
        .max_epochs(3000)
        .target_error(0.01)
        .batch_size(4) // full-batch for 4 samples
        .shuffle(false) // deterministic order
        .seed(1); // reproducible shuffling (even if shuffle=false)

    println!("Training XOR network...");
    let mut trainer = BackpropTrainer::new();
    let result = trainer
        .train(&mut network, &inputs, &targets, &config)
        .expect("training should not fail on finite inputs");

    println!("Converged:    {}", result.converged);
    println!("Final error:  {:.6}", result.final_error);
    println!("Epochs:       {}", result.epochs_trained);
    println!(
        "Error at epoch 1:    {:.4}",
        result.error_history.first().unwrap_or(&f32::NAN)
    );
    println!(
        "Error at epoch {}:  {:.4}",
        result.epochs_trained,
        result.error_history.last().unwrap_or(&f32::NAN)
    );
    println!();

    // -------------------------------------------------------------------------
    // 4. Verify predictions
    // -------------------------------------------------------------------------
    println!("Predictions after training:");
    for (input, target) in inputs.iter().zip(&targets) {
        let output = network.forward(input).expect("forward pass");
        let raw = output[0];
        let predicted = if raw > 0.0 { 1.0_f32 } else { -1.0_f32 };
        let correct = (predicted - target[0]).abs() < 0.5;
        println!(
            "  {:?} => raw={:+.4}  predicted={:+.0}  target={:+.0}  {}",
            input,
            raw,
            predicted,
            target[0],
            if correct { "ok" } else { "WRONG" },
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 5. Intermediate activations
    // -------------------------------------------------------------------------
    println!("Intermediate activations for input [0, 1]:");
    network.forward(&[0.0, 1.0]).expect("forward");
    let hidden = network.activations(0).expect("layer 0 activations");
    println!("  Hidden layer (4 neurons): {:?}", hidden);
    let output_layer = network.activations(1).expect("layer 1 activations");
    println!("  Output layer (1 neuron):  {:?}", output_layer);
    println!();

    // -------------------------------------------------------------------------
    // 6. Reproducibility check
    // -------------------------------------------------------------------------
    println!(
        "Reproducibility: two networks with the same seed should be identical after training."
    );
    let mut net_a = NetworkBuilder::new()
        .input(2)
        .hidden(4, Activation::Tanh)
        .output(1, Activation::Tanh)
        .build_with_seed(42)
        .unwrap();
    let mut net_b = NetworkBuilder::new()
        .input(2)
        .hidden(4, Activation::Tanh)
        .output(1, Activation::Tanh)
        .build_with_seed(42)
        .unwrap();

    let deterministic_config = TrainingConfig::new()
        .learning_rate(0.5)
        .momentum(0.9)
        .max_epochs(100)
        .batch_size(4)
        .shuffle(false)
        .seed(1);

    let mut trainer_a = BackpropTrainer::new();
    let mut trainer_b = BackpropTrainer::new();
    trainer_a
        .train(&mut net_a, &inputs, &targets, &deterministic_config)
        .unwrap();
    trainer_b
        .train(&mut net_b, &inputs, &targets, &deterministic_config)
        .unwrap();

    let out_a = net_a.forward(&[1.0, 0.0]).unwrap().to_vec();
    let out_b = net_b.forward(&[1.0, 0.0]).unwrap().to_vec();
    let max_diff = out_a
        .iter()
        .zip(&out_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("  Max output difference: {:.2e}  (should be ~0)", max_diff);
    assert!(
        max_diff < 1e-5,
        "training must be deterministic under same seed"
    );
    println!("  ok");
    println!();

    println!("Done.");
}
