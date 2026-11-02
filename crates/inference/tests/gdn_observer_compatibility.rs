#![cfg(feature = "gdn-state-observer")]

use std::marker::PhantomData;

fn assert_same_type<T>(_: T, _: T) {}

#[test]
fn observer_types_keep_the_family_identity_at_the_facade() {
    assert_same_type(
        PhantomData::<lattice_inference::gdn_state_observer::GdnStateObservation>,
        PhantomData::<lattice_model_qwen35::gdn_state_observer::GdnStateObservation>,
    );
    assert_same_type(
        PhantomData::<lattice_inference::gdn_state_observer::StateHealth>,
        PhantomData::<lattice_model_qwen35::gdn_state_observer::StateHealth>,
    );
}
