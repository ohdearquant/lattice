//! The inference attention facade continues to expose the compute-owned APIs.

use std::any::TypeId;

#[test]
fn attention_types_and_calls_remain_compatible() {
    macro_rules! same_type {
        ($module:ident, $name:ident) => {
            assert_eq!(
                TypeId::of::<lattice_inference::attention::$module::$name>(),
                TypeId::of::<lattice_compute::attention::$module::$name>(),
                concat!(stringify!($module), "::", stringify!($name)),
            );
        };
    }
    same_type!(differential, DiffAttnConfig);
    same_type!(differential, DiffAttnScratch);
    same_type!(differential, DiffLambdaParams);
    same_type!(native_sparse, NsaConfig);
    same_type!(native_sparse, NsaScratch);
    same_type!(native_sparse, NsaWeights);

    let _ = lattice_inference::attention::differential::apply_differential_attention;
    let _ = lattice_inference::attention::differential::compute_lambda_full;
    let _ = lattice_inference::attention::native_sparse::apply_native_sparse_attention;
    let gate = [-1.0, 0.0, 1.0];
    let mut facade = [0.5, 1.0, 2.0];
    let mut owner = facade;
    lattice_inference::attention::gated::apply_sigmoid_gate(&mut facade, &gate);
    lattice_compute::attention::gated::apply_sigmoid_gate(&mut owner, &gate);
    assert_eq!(facade, owner);
    let mut facade_fast = [1.0, 2.0, 3.0];
    let mut owner_fast = facade_fast;
    lattice_inference::attention::gated::apply_sigmoid_gate_fast(&mut facade_fast, &gate);
    lattice_compute::attention::gated::apply_sigmoid_gate_fast(&mut owner_fast, &gate);
    assert_eq!(facade_fast, owner_fast);
    let mut facade_scalar = [1.0, 2.0, 3.0];
    let mut owner_scalar = facade_scalar;
    lattice_inference::attention::gated::apply_sigmoid_gate_scalar(&mut facade_scalar, &gate);
    lattice_compute::attention::gated::apply_sigmoid_gate_scalar(&mut owner_scalar, &gate);
    assert_eq!(facade_scalar, owner_scalar);
    let mut facade_q = [0.0; 2];
    let mut facade_gate = [0.0; 2];
    let mut owner_q = [0.0; 2];
    let mut owner_gate = [0.0; 2];
    lattice_inference::attention::gated::deinterleave_q_gate(
        &[1.0, 2.0, 3.0, 4.0],
        &mut facade_q,
        &mut facade_gate,
        1,
        2,
    );
    lattice_compute::attention::gated::deinterleave_q_gate(
        &[1.0, 2.0, 3.0, 4.0],
        &mut owner_q,
        &mut owner_gate,
        1,
        2,
    );
    assert_eq!((facade_q, facade_gate), (owner_q, owner_gate));
    let mut facade_product = [0.0];
    let mut owner_product = [0.0];
    lattice_inference::forward::cpu::matmul_bt(
        &[2.0, 3.0],
        &[4.0, 5.0],
        &mut facade_product,
        1,
        2,
        1,
    );
    lattice_compute::forward::cpu::matmul_bt(&[2.0, 3.0], &[4.0, 5.0], &mut owner_product, 1, 2, 1);
    assert_eq!(facade_product, [23.0]);
    assert_eq!(facade_product, owner_product);
}
