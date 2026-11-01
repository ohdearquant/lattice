//! The forecast facade must keep exposing the owning crate's types.

use std::any::{TypeId, type_name};

use lattice_forecast as owner;
use lattice_inference::forecast as facade;

fn assert_same_type<Facade: 'static, Owner: 'static>() {
    assert_eq!(
        TypeId::of::<Facade>(),
        TypeId::of::<Owner>(),
        "{}",
        type_name::<Facade>(),
    );
}

#[test]
fn forecast_types_match_the_owner_crate() {
    assert_same_type::<facade::Chronos2, owner::Chronos2>();
    assert_same_type::<facade::Chronos2Config, owner::Chronos2Config>();
    assert_same_type::<facade::Chronos2Series, owner::Chronos2Series>();
    assert_same_type::<facade::QuantileForecast, owner::QuantileForecast>();
    assert_same_type::<facade::Reverso, owner::Reverso>();
    assert_same_type::<facade::ReversoConfig, owner::ReversoConfig>();
    assert_same_type::<facade::PredictOptions, owner::PredictOptions>();
    assert_same_type::<facade::Toto2, owner::Toto2>();
    assert_same_type::<facade::Toto2Config, owner::Toto2Config>();
    assert_same_type::<facade::Toto2Options, owner::Toto2Options>();
    assert_same_type::<facade::Toto2Variate, owner::Toto2Variate>();
}
