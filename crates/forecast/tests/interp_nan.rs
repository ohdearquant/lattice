use lattice_forecast::prep::interp;

#[test]
fn nan_query_with_multiple_points_returns_nan() {
    assert!(interp(f64::NAN, &[0.0, 1.0], &[2.0, 3.0]).is_nan());
}

#[test]
fn nan_query_with_one_point_returns_its_value() {
    assert_eq!(interp(f64::NAN, &[1.0], &[7.0]), 7.0);
}

#[test]
fn finite_queries_still_interpolate_and_extend() {
    let xp = [0.0, 1.0];
    let fp = [2.0, 4.0];
    assert_eq!(interp(-1.0, &xp, &fp), 2.0);
    assert_eq!(interp(0.0, &xp, &fp), 2.0);
    assert_eq!(interp(0.5, &xp, &fp), 3.0);
    assert_eq!(interp(1.0, &xp, &fp), 4.0);
    assert_eq!(interp(2.0, &xp, &fp), 4.0);
}
