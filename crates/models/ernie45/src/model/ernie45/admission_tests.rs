use super::tests::test_config;
use super::*;

const DOMAIN_ERROR: &str = "ernie45 config: rope_theta must be finite and positive";
const FREQUENCY_ERROR: &str =
    "ernie45 config: rope_theta produces a non-finite f32 inverse frequency";

fn config_json(cfg: &Ernie45Config) -> String {
    assert!(cfg.rope_theta.is_finite());
    serde_json::json!({
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "rope_scaling": {"mrope_section": cfg.rope_scaling.mrope_section},
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "use_bias": cfg.use_bias,
    })
    .to_string()
}

fn observe_frequencies(case: &str, cfg: &Ernie45Config) -> Vec<f32> {
    let frequencies = gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_theta, None);
    eprintln!(
        "ERNIE_THETA_FREQUENCIES {}",
        serde_json::json!({
            "case": case,
            "theta_bits": cfg.rope_theta.to_bits(),
            "head_dim": cfg.head_dim,
            "frequency_bits": frequencies.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "frequency_values": format!("{frequencies:?}"),
        })
    );
    frequencies
}

fn assert_refusal<T>(result: Result<T, InferenceError>, expected: &str, witness: &str) {
    let error = result.err();
    assert!(
        matches!(&error, Some(InferenceError::Inference(message)) if message == expected),
        "{witness}: {error:?}"
    );
}

#[test]
fn theta_domain_refuses_nonpositive_and_nonfinite_values() {
    let mut results = Vec::new();
    for (name, theta) in [
        ("zero", 0.0),
        ("negative", -1.0),
        ("nan", f64::NAN),
        ("positive_infinity", f64::INFINITY),
        ("negative_infinity", f64::NEG_INFINITY),
    ] {
        let mut cfg = test_config(128, &[16, 24, 24]);
        cfg.rope_theta = theta;
        observe_frequencies(name, &cfg);
        results.push((name, cfg.validate()));
    }
    for (name, result) in results {
        assert_refusal(
            result,
            DOMAIN_ERROR,
            &format!("theta domain admission must refuse {name}"),
        );
    }
}

#[test]
fn theta_frequency_refuses_positive_conversion_overflow() {
    let mut cfg = test_config(128, &[16, 24, 24]);
    cfg.rope_theta = f64::MIN_POSITIVE;
    let frequencies = observe_frequencies("minimum_positive", &cfg);
    assert!(cfg.rope_theta.is_finite() && cfg.rope_theta > 0.0);
    assert!(
        frequencies.iter().any(|value| !value.is_finite()),
        "the positive-theta fixture must actually overflow a derived f32 frequency"
    );
    assert_refusal(
        cfg.validate(),
        FREQUENCY_ERROR,
        "theta frequency admission must refuse positive conversion overflow",
    );
}

#[test]
fn theta_parser_refuses_finite_invalid_inputs() {
    let mut results = Vec::new();
    for (name, theta, expected) in [
        ("json_zero", 0.0, DOMAIN_ERROR),
        ("json_negative", -1.0, DOMAIN_ERROR),
        ("json_minimum_positive", f64::MIN_POSITIVE, FREQUENCY_ERROR),
    ] {
        let mut cfg = test_config(128, &[16, 24, 24]);
        cfg.rope_theta = theta;
        let text = config_json(&cfg);
        let parsed: Ernie45Config = serde_json::from_str(&text).unwrap();
        observe_frequencies(name, &parsed);
        results.push((name, expected, Ernie45Config::from_config_json_str(&text)));
    }
    for (name, expected, result) in results {
        assert_refusal(
            result,
            expected,
            &format!("JSON theta admission must refuse {name}"),
        );
    }
}

#[test]
fn theta_model_admission_precedes_weight_checks() {
    for (theta, expected) in [
        (0.0, DOMAIN_ERROR),
        (-1.0, DOMAIN_ERROR),
        (f64::NAN, DOMAIN_ERROR),
        (f64::INFINITY, DOMAIN_ERROR),
        (f64::NEG_INFINITY, DOMAIN_ERROR),
        (f64::MIN_POSITIVE, FREQUENCY_ERROR),
    ] {
        let mut cfg = test_config(128, &[16, 24, 24]);
        cfg.rope_theta = theta;
        let weights = Ernie45Weights {
            embed_tokens: Vec::new(),
            layers: Vec::new(),
            final_norm: Vec::new(),
            lm_head: Vec::new(),
        };
        assert_refusal(
            Ernie45Model::new(cfg, weights),
            expected,
            "model constructor must refuse theta before weight checks",
        );
    }
}

#[test]
fn theta_valid_controls_emit_table_bits() {
    let positions = [[0, 0, 0], [1, 2, 3], [3, 5, 7]];
    for theta in [1.0, 1_000_000.0] {
        let mut cfg = test_config(128, &[16, 24, 24]);
        cfg.rope_theta = theta;
        cfg.validate().unwrap();
        let frequencies = observe_frequencies("valid_control", &cfg);
        assert!(frequencies.iter().all(|value| value.is_finite()));
        let parsed = Ernie45Config::from_config_json_str(&config_json(&cfg)).unwrap();
        let (cos, sin) = mrope_cos_sin_table(&cfg, &positions);
        let (parsed_cos, parsed_sin) = mrope_cos_sin_table(&parsed, &positions);
        assert!(cos.iter().chain(&sin).all(|value| value.is_finite()));
        let cos_bits: Vec<_> = cos.iter().map(|value| value.to_bits()).collect();
        let sin_bits: Vec<_> = sin.iter().map(|value| value.to_bits()).collect();
        assert_eq!(
            cos_bits,
            parsed_cos.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            sin_bits,
            parsed_sin.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        eprintln!(
            "ERNIE_THETA_TABLE {}",
            serde_json::json!({
                "theta_bits": theta.to_bits(),
                "positions": positions,
                "cos_bits": cos_bits,
                "sin_bits": sin_bits,
            })
        );
    }
}

#[test]
fn theta_finite_frequency_extremes_remain_accepted() {
    let mut underflow = test_config(128, &[16, 24, 24]);
    underflow.rope_theta = 1e300;
    let frequencies = observe_frequencies("finite_zero_underflow", &underflow);
    assert!(frequencies.iter().all(|value| value.is_finite()));
    assert!(
        frequencies.contains(&0.0),
        "fixture must produce finite zero"
    );
    underflow.validate().unwrap();
    Ernie45Config::from_config_json_str(&config_json(&underflow)).unwrap();

    let mut one_frequency = test_config(2, &[1]);
    one_frequency.rope_theta = f64::MIN_POSITIVE;
    let frequencies = observe_frequencies("minimum_positive_single_frequency", &one_frequency);
    assert_eq!(frequencies, [1.0]);
    one_frequency.validate().unwrap();
    Ernie45Config::from_config_json_str(&config_json(&one_frequency)).unwrap();
}

mod angle_bound {
    use super::*;

    const ANGLE_ERROR: &str =
        "ernie45 config: rope_theta produces a non-finite f32 angle for a u32 position";

    fn config(theta: f64) -> Ernie45Config {
        let mut cfg = test_config(128, &[16, 24, 24]);
        cfg.rope_theta = theta;
        cfg
    }

    fn observed_routes(cfg: &Ernie45Config) -> (Vec<f32>, Ernie45DecodeScratch) {
        let prefill = gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_theta, None);
        let mut admitted = cfg.clone();
        admitted.rope_theta = 1.0;
        // Observe rejected configs without making fixture creation depend on that admission.
        let mut scratch = Ernie45DecodeScratch::new(&admitted, 1).unwrap();
        scratch.fill_mrope(cfg, [0; 3]);
        for (route, frequencies) in [("prefill", &prefill), ("decode", &scratch.inv_freq)] {
            assert_eq!(frequencies.len(), cfg.head_dim / 2);
            assert!(frequencies.iter().all(|f| f.is_finite() && *f >= 0.0));
            let maximum = frequencies.iter().copied().fold(0.0_f32, f32::max);
            eprintln!(
                "ERNIE_ANGLE_FREQUENCIES {}",
                serde_json::json!({
                    "route": route,
                    "theta_bits": cfg.rope_theta.to_bits(),
                    "head_dim": cfg.head_dim,
                    "frequency_bits": frequencies.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    "maximum": format!("{maximum:?}"),
                    "angle13_bits": (maximum * 13.0).to_bits(),
                    "angle14_bits": (maximum * 14.0).to_bits(),
                    "angle_u32_max_bits": (maximum * u32::MAX as f32).to_bits(),
                })
            );
        }
        (prefill, scratch)
    }

    #[test]
    fn finite_frequencies_refuse_position14_overflow() {
        let cfg = config(1e-38);
        let (prefill, mut scratch) = observed_routes(&cfg);
        for frequencies in [&prefill, &scratch.inv_freq] {
            let maximum = frequencies.iter().copied().fold(0.0_f32, f32::max);
            assert!((maximum * 13.0).is_finite());
            assert_eq!(maximum * 14.0, f32::INFINITY);
            assert_eq!(maximum * u32::MAX as f32, f32::INFINITY);
        }
        for position in [13, 14] {
            let (cos, sin) = mrope_cos_sin_table(&cfg, &[[position; 3]]);
            scratch.fill_mrope(&cfg, [position; 3]);
            assert_eq!(
                cos.iter().chain(&sin).all(|f| f.is_finite()),
                position == 13
            );
            assert_eq!(
                scratch
                    .cos
                    .iter()
                    .chain(&scratch.sin)
                    .all(|f| f.is_finite()),
                position == 13
            );
        }
        assert_refusal(
            cfg.validate(),
            ANGLE_ERROR,
            "ERNIE_ANGLE_POSITION14_REFUSAL",
        );
        eprintln!("ERNIE_ANGLE_POSITION14_COMPLETE");
    }

    #[test]
    fn finite_small_positions_do_not_waive_u32_bound() {
        let cfg = config(1e-32);
        let (prefill, scratch) = observed_routes(&cfg);
        for frequencies in [&prefill, &scratch.inv_freq] {
            let maximum = frequencies.iter().copied().fold(0.0_f32, f32::max);
            assert!((maximum * 14.0).is_finite());
            assert_eq!(maximum * u32::MAX as f32, f32::INFINITY);
        }
        assert_refusal(cfg.validate(), ANGLE_ERROR, "ERNIE_ANGLE_U32_REFUSAL");
        eprintln!("ERNIE_ANGLE_U32_COMPLETE");
    }

    #[test]
    fn parser_refuses_angle_overflow() {
        let cfg = config(1e-38);
        let text = config_json(&cfg);
        let parsed: Ernie45Config = serde_json::from_str(&text).unwrap();
        assert!(parsed.rope_theta.is_finite() && parsed.rope_theta > 0.0);
        observed_routes(&parsed);
        assert_refusal(
            Ernie45Config::from_config_json_str(&text),
            ANGLE_ERROR,
            "ERNIE_ANGLE_PARSER_REFUSAL",
        );
        eprintln!("ERNIE_ANGLE_PARSER_COMPLETE");
    }

    #[test]
    fn model_refuses_angles_before_weights() {
        let cfg = config(1e-38);
        let weights = Ernie45Weights {
            embed_tokens: Vec::new(),
            layers: Vec::new(),
            final_norm: Vec::new(),
            lm_head: Vec::new(),
        };
        assert_refusal(
            Ernie45Model::new(cfg, weights),
            ANGLE_ERROR,
            "ERNIE_ANGLE_MODEL_REFUSAL",
        );
        eprintln!("ERNIE_ANGLE_MODEL_COMPLETE");
    }

    #[test]
    fn accepted_controls_preserve_both_tables() {
        let positions = [[0; 3], [13, 14, 15], [u32::MAX; 3]];
        for theta in [0.5, 1.0, 1_000_000.0] {
            let cfg = config(theta);
            cfg.validate().unwrap();
            Ernie45Config::from_config_json_str(&config_json(&cfg)).unwrap();
            let (prefill, mut scratch) = observed_routes(&cfg);
            for frequencies in [&prefill, &scratch.inv_freq] {
                assert!(
                    frequencies
                        .iter()
                        .all(|f| (*f * u32::MAX as f32).is_finite())
                );
            }
            let (cos, sin) = mrope_cos_sin_table(&cfg, &positions);
            let mut decode_cos = Vec::new();
            let mut decode_sin = Vec::new();
            for position in positions {
                scratch.fill_mrope(&cfg, position);
                decode_cos.extend_from_slice(&scratch.cos);
                decode_sin.extend_from_slice(&scratch.sin);
            }
            for (route, cos, sin) in [("prefill", cos, sin), ("decode", decode_cos, decode_sin)] {
                assert_eq!(cos.len(), positions.len() * cfg.head_dim);
                assert_eq!(sin.len(), cos.len());
                assert!(cos.iter().chain(&sin).all(|f| f.is_finite()));
                eprintln!(
                    "ERNIE_ANGLE_TABLE {}",
                    serde_json::json!({
                        "route": route,
                        "theta_bits": theta.to_bits(),
                        "positions": positions,
                        "cos_bits": cos.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                        "sin_bits": sin.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    })
                );
            }
        }
        eprintln!("ERNIE_ANGLE_CONTROLS_COMPLETE");
    }
}
