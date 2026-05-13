//! Property-based tests for IntentLabels softmax normalization
//!
//! Issue #449: proptest verifying IntentLabels::softmax_normalize() produces
//! probabilities summing to 1.0

use lattice_tune::data::IntentLabels;
use proptest::prelude::*;

proptest! {
    // Test that softmax_normalize produces probabilities summing to 1.0
    #[test]
    fn test_softmax_normalize_sums_to_one(
        continuation in -10.0f32..10.0,
        topic_shift in -10.0f32..10.0,
        explicit_query in -10.0f32..10.0,
        person_lookup in -10.0f32..10.0,
        health_check in -10.0f32..10.0,
        task_status in -10.0f32..10.0,
    ) {
        let mut labels = IntentLabels {
            continuation,
            topic_shift,
            explicit_query,
            person_lookup,
            health_check,
            task_status,
        };

        labels.softmax_normalize().unwrap();

        // Get all probabilities
        let probs = labels.to_vec();

        // Sum should be approximately 1.0
        let sum: f32 = probs.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "Sum {} is not approximately 1.0 (diff: {})",
            sum,
            (sum - 1.0).abs()
        );

        // All probabilities should be in [0, 1]
        for (i, &p) in probs.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&p),
                "Probability {} at index {} is not in [0, 1]",
                p,
                i
            );
        }
    }

    // Test softmax with extreme values
    #[test]
    fn test_softmax_extreme_values(
        large_val in 50.0f32..100.0,
        small_val in -100.0f32..-50.0,
    ) {
        let mut labels = IntentLabels {
            continuation: large_val,
            topic_shift: 0.0,
            explicit_query: small_val,
            person_lookup: 0.0,
            health_check: small_val,
            task_status: 0.0,
        };

        labels.softmax_normalize().unwrap();

        let probs = labels.to_vec();
        let sum: f32 = probs.iter().sum();

        // Even with extreme values, sum should be ~1.0
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "Sum {} not close to 1.0 with extreme values",
            sum
        );

        // All values should be valid probabilities
        for &p in &probs {
            prop_assert!(
                !p.is_nan() && !p.is_infinite(),
                "Invalid probability value: {}",
                p
            );
        }
    }

    // Test softmax preserves ordering (largest input = largest output)
    #[test]
    fn test_softmax_preserves_ordering(
        base in 0.0f32..5.0,
        delta1 in 0.1f32..5.0,
        delta2 in 0.1f32..5.0,
        delta3 in 0.1f32..5.0,
    ) {
        let mut labels = IntentLabels {
            continuation: base + delta1 + delta2 + delta3, // largest
            topic_shift: base + delta1 + delta2,
            explicit_query: base + delta1,
            person_lookup: base,
            health_check: base - 0.5,
            task_status: base - 1.0, // smallest
        };

        labels.softmax_normalize().unwrap();

        // continuation should have highest probability
        let (dominant_name, _) = labels.dominant();
        prop_assert_eq!(
            dominant_name,
            "continuation",
            "Expected continuation to be dominant"
        );
    }

    // Test softmax with uniform inputs
    #[test]
    fn test_softmax_uniform_inputs(value in -5.0f32..5.0) {
        let mut labels = IntentLabels {
            continuation: value,
            topic_shift: value,
            explicit_query: value,
            person_lookup: value,
            health_check: value,
            task_status: value,
        };

        labels.softmax_normalize().unwrap();

        let probs = labels.to_vec();

        // With uniform inputs, all probabilities should be roughly equal (1/6)
        let expected = 1.0 / 6.0;
        for &p in &probs {
            prop_assert!(
                (p - expected).abs() < 1e-4,
                "Probability {} not close to expected {} for uniform inputs",
                p,
                expected
            );
        }
    }

    // Test that softmax produces valid probability distribution
    #[test]
    fn test_softmax_valid_probability_distribution(
        vals in prop::array::uniform6(-20.0f32..20.0),
    ) {
        let mut labels = IntentLabels {
            continuation: vals[0],
            topic_shift: vals[1],
            explicit_query: vals[2],
            person_lookup: vals[3],
            health_check: vals[4],
            task_status: vals[5],
        };

        labels.softmax_normalize().unwrap();

        let probs = labels.to_vec();

        // Sum to 1
        let sum: f32 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "Sum {} != 1.0", sum);

        // All non-negative
        for &p in &probs {
            prop_assert!(p >= 0.0, "Negative probability: {}", p);
        }

        // No NaN/Inf
        for &p in &probs {
            prop_assert!(!p.is_nan(), "NaN in probabilities");
            prop_assert!(!p.is_infinite(), "Inf in probabilities");
        }
    }
}

// Test softmax with zeros (regular test)
#[test]
fn test_softmax_with_zeros() {
    let mut labels = IntentLabels::default(); // all zeros

    labels.softmax_normalize().unwrap();

    let probs = labels.to_vec();
    let sum: f32 = probs.iter().sum();

    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Sum {sum} is not approximately 1.0 with all-zero inputs"
    );
}

#[test]
fn test_softmax_basic() {
    let mut labels = IntentLabels {
        continuation: 2.0,
        topic_shift: 1.0,
        explicit_query: 0.5,
        person_lookup: 0.0,
        health_check: 0.0,
        task_status: 0.0,
    };

    labels.softmax_normalize().unwrap();

    let sum: f32 = labels.to_vec().iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
}
