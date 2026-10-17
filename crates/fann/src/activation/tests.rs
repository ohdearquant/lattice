use super::*;

const EPSILON: f32 = 1e-5;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

#[test]
fn test_linear() {
    let act = Activation::Linear;
    assert!(approx_eq(act.forward(0.5), 0.5));
    assert!(approx_eq(act.forward(-1.0), -1.0));
    assert!(approx_eq(act.derivative(0.5), 1.0));
}

#[test]
fn test_sigmoid() {
    let act = Activation::Sigmoid;
    assert!(approx_eq(act.forward(0.0), 0.5));
    assert!(act.forward(10.0) > 0.99);
    assert!(act.forward(-10.0) < 0.01);
    // Derivative at output 0.5
    assert!(approx_eq(act.derivative(0.5), 0.25));
}

#[test]
fn test_tanh() {
    let act = Activation::Tanh;
    assert!(approx_eq(act.forward(0.0), 0.0));
    assert!(act.forward(3.0) > 0.99);
    assert!(act.forward(-3.0) < -0.99);
    // Derivative at output 0
    assert!(approx_eq(act.derivative(0.0), 1.0));
}

#[test]
fn test_relu() {
    let act = Activation::ReLU;
    assert!(approx_eq(act.forward(0.5), 0.5));
    assert!(approx_eq(act.forward(-0.5), 0.0));
    assert!(approx_eq(act.forward(0.0), 0.0));
    assert!(approx_eq(act.derivative(0.5), 1.0));
    assert!(approx_eq(act.derivative(0.0), 0.0));
}

#[test]
fn test_leaky_relu() {
    let alpha = 0.1;
    let act = Activation::LeakyReLU(alpha);
    assert!(approx_eq(act.forward(0.5), 0.5));
    assert!(approx_eq(act.forward(-0.5), -0.05));
    assert!(approx_eq(act.derivative(0.5), 1.0));
    assert!(approx_eq(act.derivative(-0.5), alpha));
}

#[test]
fn test_softmax_batch() {
    let act = Activation::Softmax;
    let mut values = vec![1.0, 2.0, 3.0];
    act.forward_batch(&mut values);

    // Sum should be 1.0
    let sum: f32 = values.iter().sum();
    assert!(approx_eq(sum, 1.0));

    // Values should be in increasing order (since input was increasing)
    assert!(values[0] < values[1]);
    assert!(values[1] < values[2]);
}

#[test]
fn test_softmax_numerical_stability() {
    let act = Activation::Softmax;
    // Large values that would overflow without stability fix
    let mut values = vec![1000.0, 1001.0, 1002.0];
    act.forward_batch(&mut values);

    let sum: f32 = values.iter().sum();
    assert!(approx_eq(sum, 1.0));
    assert!(values.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_forward_batch_consistency() {
    let activations = vec![
        Activation::Linear,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::ReLU,
        Activation::LeakyReLU(0.1),
    ];

    for act in activations {
        let values = vec![-1.0, 0.0, 0.5, 1.0];
        let single_results: Vec<f32> = values.iter().map(|&v| act.forward(v)).collect();

        let mut batch_values = values.clone();
        act.forward_batch(&mut batch_values);

        for (single, batch) in single_results.iter().zip(batch_values.iter()) {
            assert!(
                approx_eq(*single, *batch),
                "{act:?}: single={single}, batch={batch}"
            );
        }
    }
}

#[test]
fn test_is_bounded() {
    assert!(!Activation::Linear.is_bounded());
    assert!(Activation::Sigmoid.is_bounded());
    assert!(Activation::Tanh.is_bounded());
    assert!(!Activation::ReLU.is_bounded());
    assert!(!Activation::LeakyReLU(0.1).is_bounded());
    assert!(Activation::Softmax.is_bounded());
}

#[test]
fn test_default_is_relu() {
    assert_eq!(Activation::default(), Activation::ReLU);
}
