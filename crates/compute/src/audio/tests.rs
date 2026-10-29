use super::*;

#[test]
fn silence_has_expected_shape_and_log_floor() {
    let plan = WhisperLogMelPlan::new();
    let result = plan.extract(&vec![0.0; SAMPLE_RATE], SAMPLE_RATE).unwrap();
    assert_eq!(result.frames, 100);
    assert_eq!(result.values.len(), 128 * 100);
    assert!(result.values.iter().all(|value| *value == -1.5));
}

#[test]
fn input_validation_and_partial_frame() {
    let plan = WhisperLogMelPlan::new();
    assert!(plan.extract(&[0.0; 200], SAMPLE_RATE).is_err());
    assert!(plan.extract(&[0.0; 201], 8_000).is_err());
    let mut samples = vec![0.0; 321];
    samples[10] = f32::NAN;
    assert!(plan.extract(&samples, SAMPLE_RATE).is_err());
    samples[10] = f32::INFINITY;
    assert!(plan.extract(&samples, SAMPLE_RATE).is_err());
    samples[10] = 0.0;
    let result = plan.extract(&samples, SAMPLE_RATE).unwrap();
    assert_eq!(result.frames, 2);
    assert_eq!(result.values.len(), 256);
}

#[test]
fn slaney_conversions_and_periodic_hann_anchors() {
    let plan = WhisperLogMelPlan::new();
    assert_eq!(hz_to_mel(1000.0), 15.0);
    assert_eq!(mel_to_hz(15.0), 1000.0);
    assert_eq!(plan.window[0], 0.0);
    assert!((plan.window[200] - 1.0).abs() < 1e-15);
    assert!(plan.window[399] > 0.0);
    assert!(
        plan.filters
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
}

#[test]
fn reflection_and_dropped_last_frame_follow_short_boundaries() {
    let len = 201;
    assert_eq!(reflect_index(0, len), 200);
    assert_eq!(reflect_index(199, len), 1);
    assert_eq!(reflect_index(200, len), 0);
    assert_eq!(reflect_index(400, len), 200);
    assert_eq!(reflect_index(401, len), 199);
    let plan = WhisperLogMelPlan::new();
    for (samples, frames) in [(201, 1), (319, 1), (320, 2), (321, 2)] {
        let mut input = vec![0.0; samples];
        input[0] = 1.0;
        input[samples - 1] = -1.0;
        let result = plan.extract(&input, SAMPLE_RATE).unwrap();
        assert_eq!(result.frames, frames);
        assert!(result.values.iter().all(|value| value.is_finite()));
    }
}
