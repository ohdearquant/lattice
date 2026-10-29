//! The forecast facade must keep exposing the owning crate's types.

use std::any::TypeId;

#[test]
fn forecast_types_match_the_owner_crate() {
    macro_rules! same_type {
        ($name:ident) => {
            assert_eq!(
                TypeId::of::<lattice_inference::forecast::$name>(),
                TypeId::of::<lattice_forecast::$name>(),
                stringify!($name),
            );
        };
    }
    same_type!(Chronos2);
    same_type!(Chronos2Config);
    same_type!(Chronos2Series);
    same_type!(QuantileForecast);
    same_type!(Reverso);
    same_type!(ReversoConfig);
    same_type!(PredictOptions);
    same_type!(Toto2);
    same_type!(Toto2Config);
    same_type!(Toto2Options);
    same_type!(Toto2Variate);
}
