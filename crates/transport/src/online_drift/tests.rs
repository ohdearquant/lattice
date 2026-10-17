use super::*;

#[test]
fn window_size_above_max_is_clamped() {
    let config = OnlineDriftConfig {
        window_size: usize::MAX,
        ..Default::default()
    };
    assert_eq!(
        config.normalized().window_size,
        MAX_ONLINE_DRIFT_WINDOW_SIZE
    );
}

#[test]
fn window_size_below_min_is_raised() {
    let config = OnlineDriftConfig {
        window_size: 1,
        ..Default::default()
    };
    assert_eq!(
        config.normalized().window_size,
        MIN_ONLINE_DRIFT_WINDOW_SIZE
    );
}

// Deterministic linear congruential generator so tests don't need `rand`.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map high 32 bits to [0, 1).
        (self.state >> 32) as f32 / u32::MAX as f32
    }

    // Sample from approximate N(mean, 1) using Box-Muller (two uniforms -> one normal).
    fn next_normal(&mut self, mean: f32) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + z
    }
}

fn make_vector(rng: &mut Lcg, dim: usize, mean: f32) -> Vec<f32> {
    (0..dim).map(|_| rng.next_normal(mean)).collect()
}

#[test]
fn online_drift_warms_until_window_full() {
    let config = OnlineDriftConfig {
        window_size: 128,
        check_interval: 1,
        threshold: 100.0, // never fires drift
        ..Default::default()
    };
    let mut detector = OnlineDriftDetector::new(config);
    let mut rng = Lcg::new(42);

    for i in 1..128 {
        let signal = detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap();
        assert!(
            matches!(
                signal,
                OnlineDriftSignal::Warming {
                    samples_seen,
                    window_size: 128
                } if samples_seen == i
            ),
            "expected Warming at sample {i}, got {signal:?}"
        );
    }
    assert_eq!(detector.samples_seen(), 127);
}

#[test]
fn online_drift_computes_when_window_fills() {
    // With check_interval=1 and a high threshold, sample 128 fills both windows
    // with identical data, which should yield near-zero divergence.
    let config = OnlineDriftConfig {
        window_size: 128,
        check_interval: 1,
        threshold: 100.0,
        ..Default::default()
    };
    let mut detector = OnlineDriftDetector::new(config);
    let mut rng = Lcg::new(99);

    let mut last_signal = None;
    for _ in 0..128 {
        last_signal = Some(detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap());
    }

    let signal = last_signal.unwrap();
    match signal {
        OnlineDriftSignal::Stable {
            divergence,
            window_pos: 128,
        } => {
            assert!(
                divergence <= 1e-2,
                "expected near-zero divergence for identical windows, got {divergence}"
            );
        }
        other => panic!("expected Stable at sample 128, got {other:?}"),
    }
    assert!(detector.last_divergence().is_some());
}

#[test]
fn online_drift_skips_between_check_intervals() {
    let config = OnlineDriftConfig {
        window_size: 128,
        check_interval: 8,
        threshold: 100.0,
        ..Default::default()
    };
    let mut detector = OnlineDriftDetector::new(config);
    let mut rng = Lcg::new(7);

    // Feed 128 samples — fills both windows; sample 128 is divisible by 8 so it
    // triggers a compute. Sample 129 is not divisible by 8 so it must be Skipped.
    for _ in 0..128 {
        detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap();
    }
    let at_128 = detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap();
    // sample 129 (not on interval 8)
    let at_129 = at_128; // we already consumed it; redo:
    // Reset and replay carefully.
    let mut detector2 = OnlineDriftDetector::new(OnlineDriftConfig {
        window_size: 128,
        check_interval: 8,
        threshold: 100.0,
        ..Default::default()
    });
    let mut rng2 = Lcg::new(7);
    for _ in 0..128 {
        detector2.observe(make_vector(&mut rng2, 4, 0.0)).unwrap();
    }
    // sample 129 — check_interval=8, 129 % 8 = 1 ≠ 0 → Skipped
    let sig_129 = detector2.observe(make_vector(&mut rng2, 4, 0.0)).unwrap();
    assert!(
        matches!(
            sig_129,
            OnlineDriftSignal::Skipped {
                samples_seen: 129,
                next_check_in: 7
            }
        ),
        "expected Skipped at 129 with next_check_in=7, got {sig_129:?}"
    );
    // Verify sample 128 was Stable or Drift (not Warming or Skipped).
    let mut detector3 = OnlineDriftDetector::new(OnlineDriftConfig {
        window_size: 128,
        check_interval: 8,
        threshold: 100.0,
        ..Default::default()
    });
    let mut rng3 = Lcg::new(7);
    for _ in 0..127 {
        detector3.observe(make_vector(&mut rng3, 4, 0.0)).unwrap();
    }
    // sample 128 — 128 % 8 == 0 → compute
    let sig_128 = detector3.observe(make_vector(&mut rng3, 4, 0.0)).unwrap();
    assert!(
        matches!(
            sig_128,
            OnlineDriftSignal::Stable { .. } | OnlineDriftSignal::Drift { .. }
        ),
        "expected compute at sample 128, got {sig_128:?}"
    );
    let _ = at_129; // silence unused warning
}

#[test]
fn online_drift_detects_gaussian_shift() {
    // Feed 128 reference samples near 0.0, then 128 shifted samples near 4.0.
    // With a low threshold, a Drift signal must appear after the shift.
    let config = OnlineDriftConfig {
        window_size: 128,
        check_interval: 1,
        threshold: 0.1,
        ..Default::default()
    };
    let mut detector = OnlineDriftDetector::new(config);
    let mut rng = Lcg::new(2025);

    // Warm up with reference near 0.
    for _ in 0..128 {
        let sig = detector.observe(make_vector(&mut rng, 8, 0.0)).unwrap();
        // These may be Warming or Stable; no assertion needed during warm-up.
        let _ = sig;
    }

    // Feed shifted samples; at least one Drift should appear.
    let mut found_drift = false;
    for _ in 0..128 {
        let sig = detector.observe(make_vector(&mut rng, 8, 4.0)).unwrap();
        if let OnlineDriftSignal::Drift {
            divergence,
            window_pos,
        } = sig
        {
            assert!(divergence > 0.1, "drift divergence should exceed threshold");
            // window_pos is samples_seen at detection time; must be > 128 since windows
            // need to be full first, and we're in the shifted batch.
            assert!(
                window_pos > 128,
                "window_pos must be after warm-up: got {window_pos}"
            );
            found_drift = true;
            break;
        }
    }
    assert!(
        found_drift,
        "expected at least one Drift signal after distribution shift"
    );
}

#[test]
fn online_drift_zero_divergence_identical_distributions() {
    // After reset_reference, feeding the same sequence should give near-zero divergence.
    let config = OnlineDriftConfig {
        window_size: 128,
        check_interval: 1,
        threshold: 100.0,
        ..Default::default()
    };
    let mut detector = OnlineDriftDetector::new(config);
    let mut rng = Lcg::new(55);

    // Collect 128 fixed vectors.
    let vecs: Vec<Vec<f32>> = (0..128).map(|_| make_vector(&mut rng, 4, 0.0)).collect();

    // Fill detector with the fixed sequence.
    for v in &vecs {
        detector.observe(v.clone()).unwrap();
    }

    // Promote current window to reference, then replay identical sequence.
    detector.reset_reference();

    let mut last_signal = None;
    for v in &vecs {
        last_signal = Some(detector.observe(v.clone()).unwrap());
    }

    match last_signal.unwrap() {
        OnlineDriftSignal::Stable { divergence, .. } => {
            assert!(
                divergence <= 1e-4,
                "expected near-zero divergence for identical windows after reset, got {divergence}"
            );
        }
        other => panic!("expected Stable for identical distributions, got {other:?}"),
    }
}

#[test]
fn online_drift_reproducible_with_fixed_sequence() {
    // Two detectors with identical config and same input sequence must produce identical signals.
    let config = OnlineDriftConfig {
        window_size: 128,
        check_interval: 8,
        threshold: 0.05,
        ..Default::default()
    };

    let mut rng = Lcg::new(1337);
    let sequence: Vec<Vec<f32>> = (0..256).map(|_| make_vector(&mut rng, 8, 0.0)).collect();

    let mut det_a = OnlineDriftDetector::new(config.clone());
    let mut det_b = OnlineDriftDetector::new(config);

    let signals_a: Vec<OnlineDriftSignal> = sequence
        .iter()
        .map(|v| det_a.observe(v.clone()).unwrap())
        .collect();
    let signals_b: Vec<OnlineDriftSignal> = sequence
        .iter()
        .map(|v| det_b.observe(v.clone()).unwrap())
        .collect();

    assert_eq!(signals_a.len(), signals_b.len());
    for (i, (a, b)) in signals_a.iter().zip(&signals_b).enumerate() {
        match (a, b) {
            (
                OnlineDriftSignal::Stable {
                    divergence: da,
                    window_pos: pa,
                },
                OnlineDriftSignal::Stable {
                    divergence: db,
                    window_pos: pb,
                },
            ) => {
                assert_eq!(pa, pb, "window_pos mismatch at index {i}");
                assert!(
                    (da - db).abs() <= 1e-6,
                    "divergence mismatch at index {i}: {da} vs {db}"
                );
            }
            (
                OnlineDriftSignal::Drift {
                    divergence: da,
                    window_pos: pa,
                },
                OnlineDriftSignal::Drift {
                    divergence: db,
                    window_pos: pb,
                },
            ) => {
                assert_eq!(pa, pb, "window_pos mismatch at index {i}");
                assert!(
                    (da - db).abs() <= 1e-6,
                    "divergence mismatch at index {i}: {da} vs {db}"
                );
            }
            _ => assert_eq!(a, b, "signal variant mismatch at index {i}: {a:?} vs {b:?}"),
        }
    }
}
