use super::*;

fn direct_dft_log_mel(plan: &WhisperLogMelPlan, samples: &[f32]) -> Vec<f32> {
    let frames = samples.len() / HOP;
    let mut cosine = Vec::with_capacity(FREQUENCIES * FFT_SIZE);
    let mut sine = Vec::with_capacity(FREQUENCIES * FFT_SIZE);
    for k in 0..FREQUENCIES {
        for i in 0..FFT_SIZE {
            let (sin, cos) = (2.0 * PI * (k * i) as f64 / FFT_SIZE as f64).sin_cos();
            cosine.push(cos);
            sine.push(sin);
        }
    }

    let mut mel = vec![0.0f64; MELS * frames];
    let mut windowed = [0.0f64; FFT_SIZE];
    let mut power = [0.0f64; FREQUENCIES];
    for frame in 0..frames {
        let start = frame * HOP;
        for (i, sample) in windowed.iter_mut().enumerate() {
            *sample = f64::from(samples[reflect_index(start + i, samples.len())]) * plan.window[i];
        }
        for (k, value) in power.iter_mut().enumerate() {
            let mut re = 0.0;
            let mut im = 0.0;
            let base = k * FFT_SIZE;
            for (i, &sample) in windowed.iter().enumerate() {
                re += sample * cosine[base + i];
                im += sample * sine[base + i];
            }
            *value = re * re + im * im;
        }
        for m in 0..MELS {
            let energy: f64 = power
                .iter()
                .zip(&plan.filters[m * FREQUENCIES..(m + 1) * FREQUENCIES])
                .map(|(value, weight)| value * weight)
                .sum();
            mel[m * frames + frame] = energy.max(1e-10).log10();
        }
    }
    let max = mel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    mel.into_iter()
        .map(|value| ((value.max(max - 8.0) + 4.0) / 4.0) as f32)
        .collect()
}

#[test]
fn complete_log_mel_matches_direct_dft_for_one_and_thirty_seconds() {
    const MAX_ABS_ERROR: f32 = 1e-5;
    let plan = WhisperLogMelPlan::new();
    for seconds in [1usize, 30] {
        let samples: Vec<f32> = (0..seconds * SAMPLE_RATE)
            .map(|i| {
                if i % SAMPLE_RATE < SAMPLE_RATE / 4 {
                    0.0
                } else {
                    let t = i as f64 / SAMPLE_RATE as f64;
                    (0.4 * (2.0 * PI * 440.0 * t).sin() + 0.15 * (2.0 * PI * 997.0 * t).cos())
                        as f32
                }
            })
            .collect();
        let expected = direct_dft_log_mel(&plan, &samples);
        let actual = plan.extract(&samples, SAMPLE_RATE).unwrap();
        assert_eq!(actual.frames, seconds * SAMPLE_RATE / HOP);
        assert_eq!(actual.values.len(), expected.len());
        let mut max_error = 0.0f32;
        for (index, (&observed, &reference)) in actual.values.iter().zip(&expected).enumerate() {
            let error = (observed - reference).abs();
            assert!(
                error <= MAX_ABS_ERROR,
                "seconds={seconds}, index={index}, observed={observed}, reference={reference}, error={error}"
            );
            max_error = max_error.max(error);
        }
        eprintln!("seconds={seconds}, max_abs_error={max_error}");
    }
}

#[test]
fn frame_spectrum_matches_direct_dft_for_all_real_bins() {
    let plan = WhisperLogMelPlan::new();
    let cases: [[f64; FFT_SIZE]; 9] = [
        [0.0; FFT_SIZE],
        std::array::from_fn(|i| if i == 0 { 1.0 } else { 0.0 }),
        std::array::from_fn(|i| if i == FFT_SIZE - 1 { 1.0 } else { 0.0 }),
        [1.0; FFT_SIZE],
        std::array::from_fn(|i| if i.is_multiple_of(2) { 1.0 } else { -1.0 }),
        std::array::from_fn(|i| (2.0 * PI * 7.0 * i as f64 / FFT_SIZE as f64).cos()),
        std::array::from_fn(|i| (2.0 * PI * 19.0 * i as f64 / FFT_SIZE as f64).sin()),
        std::array::from_fn(|i| ((i * 37 % 97) as f64 - 48.0) / 48.0),
        std::array::from_fn(|i| {
            let t = 2.0 * PI * i as f64 / FFT_SIZE as f64;
            (7.0 * t).cos() - (7.0 * t).sin() + 1e-7 * (199.0 * t).cos()
        }),
    ];
    for (case, input) in cases.iter().enumerate() {
        let mut actual = [(0.0, 0.0); FFT_SIZE];
        plan.frame_spectrum(input, &mut actual);
        for (k, &(actual_re, actual_im)) in actual.iter().take(FREQUENCIES).enumerate() {
            let (mut expected_re, mut expected_im) = (0.0, 0.0);
            for (i, &sample) in input.iter().enumerate() {
                let angle = 2.0 * PI * (k * i) as f64 / FFT_SIZE as f64;
                let (sin, cos) = angle.sin_cos();
                expected_re += sample * cos;
                expected_im += sample * sin;
            }
            for (part, observed, expected) in [
                ("real", actual_re, expected_re),
                ("imaginary", actual_im, expected_im),
            ] {
                let tolerance = 1e-9 * (1.0 + expected.abs());
                assert!(
                    (observed - expected).abs() <= tolerance,
                    "case={case}, bin={k}, part={part}, observed={observed}, expected={expected}, tolerance={tolerance}"
                );
            }
        }
    }
}

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

#[test]
fn quiet_frames_use_the_global_eight_decade_floor() {
    let plan = WhisperLogMelPlan::new();
    let mut samples = vec![0.0_f32; SAMPLE_RATE];
    for (i, sample) in samples[..SAMPLE_RATE / 2].iter_mut().enumerate() {
        *sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin();
    }
    let output = plan.extract(&samples, SAMPLE_RATE).unwrap();
    assert_eq!(output.frames, 100);
    let peak = output
        .values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        peak > 1.0,
        "fixture must place the global floor above the raw silence floor: {peak}"
    );
    let silence_floor = output.values[80];
    assert!(
        (silence_floor - (peak - 2.0)).abs() <= 2.0 * f32::EPSILON,
        "silent frames must be clamped to the global log maximum minus eight decades: peak={peak}, floor={silence_floor}"
    );
    for channel in 0..MELS {
        for frame in 80..100 {
            assert_eq!(
                output.values[channel * output.frames + frame].to_bits(),
                silence_floor.to_bits()
            );
        }
    }
}
