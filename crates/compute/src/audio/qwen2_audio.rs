//! A bounded Qwen2-Audio input processor for mono 16 kHz PCM.
//!
//! This produces features and a feature mask, not audio embeddings or text.

use super::{WhisperLogMel, WhisperLogMelPlan};
use lattice_inference_core::InferenceError;

const RATE: usize = 16_000;
const SAMPLES: usize = 480_000;
const HOP: usize = 160;
const FRAMES: usize = SAMPLES / HOP;
const MEL_BINS: usize = 128;

/// **Unstable**: model inputs for one bounded Qwen2-Audio clip.
pub struct Qwen2AudioFeatures {
    /// Mel-major `[128, 3000]` log-mel values.
    pub features: Vec<f32>,
    /// One for each real input frame, zero for padded frames.
    pub feature_attention_mask: Vec<u8>,
    /// Number of audio soft-token positions after convolution and pooling.
    pub audio_tokens: usize,
}

/// Process one nonempty mono 16 kHz clip, with a maximum duration of 30 seconds.
///
/// Padding is applied to PCM before feature extraction; oversized clips are
/// refused instead of truncated. The plan can be reused across calls.
///
/// # Errors
///
/// Rejects an empty or oversized clip, incorrect sample rate, nonfinite input,
/// allocation failure, or a failed/inconsistent frontend result.
pub fn process(
    plan: &WhisperLogMelPlan,
    samples: &[f32],
    sample_rate: usize,
) -> Result<Qwen2AudioFeatures, InferenceError> {
    process_with(samples, sample_rate, |padded| plan.extract(padded, RATE))
}

fn process_with(
    samples: &[f32],
    sample_rate: usize,
    extract: impl FnOnce(&[f32]) -> Result<WhisperLogMel, InferenceError>,
) -> Result<Qwen2AudioFeatures, InferenceError> {
    let invalid = |message: &str| InferenceError::InvalidInput(message.to_owned());
    if sample_rate != RATE
        || samples.is_empty()
        || samples.len() > SAMPLES
        || samples.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "Qwen2-Audio requires nonempty finite mono 16 kHz PCM of at most 480000 samples",
        ));
    }
    let mut padded = Vec::new();
    padded
        .try_reserve_exact(SAMPLES)
        .map_err(|_| invalid("Qwen2-Audio PCM allocation failed"))?;
    padded.extend_from_slice(samples);
    padded.resize(SAMPLES, 0.0);
    let mel = extract(&padded)?;
    if mel.frames != FRAMES
        || mel.values.len() != MEL_BINS * FRAMES
        || mel.values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("Qwen2-Audio frontend produced invalid features"));
    }
    let valid_frames = samples.len().div_ceil(HOP);
    let mut mask = Vec::new();
    mask.try_reserve_exact(FRAMES)
        .map_err(|_| invalid("Qwen2-Audio mask allocation failed"))?;
    mask.resize(valid_frames, 1);
    mask.resize(FRAMES, 0);
    // HF stride-2 convolution followed by stride-2 pooling.
    let audio_tokens = valid_frames.div_ceil(2) / 2;
    Ok(Qwen2AudioFeatures {
        features: mel.values,
        feature_attention_mask: mask,
        audio_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(samples: &[f32]) -> Result<WhisperLogMel, InferenceError> {
        assert_eq!(samples.len(), SAMPLES);
        assert_eq!(samples[0], 0.5);
        assert_eq!(samples[1], -0.25);
        assert!(samples[2..].iter().all(|x| *x == 0.0));
        Ok(WhisperLogMel {
            values: vec![0.0; MEL_BINS * FRAMES],
            frames: FRAMES,
        })
    }

    #[test]
    fn pads_samples_before_extraction_and_derives_mask() {
        let out = process_with(&[0.5, -0.25], RATE, fake).unwrap();
        assert_eq!(out.features.len(), MEL_BINS * FRAMES);
        assert_eq!(out.feature_attention_mask.len(), FRAMES);
        assert_eq!(out.feature_attention_mask[0], 1);
        assert!(out.feature_attention_mask[1..].iter().all(|v| *v == 0));
        assert_eq!(out.audio_tokens, 0);
    }

    #[test]
    fn mask_and_token_boundaries_match_convolution_pooling() {
        for (samples, valid, tokens) in [
            (1, 1, 0),
            (159, 1, 0),
            (160, 1, 0),
            (161, 2, 0),
            (320, 2, 0),
            (321, 3, 1),
            (480_000, 3000, 750),
        ] {
            let input = vec![0.5; samples];
            let out = process_with(&input, RATE, |pcm| {
                assert_eq!(&pcm[..samples], input.as_slice());
                assert!(pcm[samples..].iter().all(|x| *x == 0.0));
                Ok(WhisperLogMel {
                    values: vec![0.0; MEL_BINS * FRAMES],
                    frames: FRAMES,
                })
            })
            .unwrap();
            assert_eq!(
                out.feature_attention_mask
                    .iter()
                    .map(|&v| usize::from(v))
                    .sum::<usize>(),
                valid
            );
            assert!(out.feature_attention_mask[..valid].iter().all(|&v| v == 1));
            assert!(out.feature_attention_mask[valid..].iter().all(|&v| v == 0));
            assert_eq!(out.audio_tokens, tokens);
        }
    }

    #[test]
    fn rejects_invalid_input_before_frontend() {
        for (input, rate) in [
            (vec![], RATE),
            (vec![1.0], 8_000),
            (vec![f32::NAN], RATE),
            (vec![f32::INFINITY], RATE),
            (vec![0.0; SAMPLES + 1], RATE),
        ] {
            assert!(process_with(&input, rate, |_| panic!("must not extract")).is_err());
        }
    }

    #[test]
    fn rejects_frontend_failure_and_bad_shapes() {
        assert!(
            process_with(&[1.0], RATE, |_| Err(InferenceError::InvalidInput(
                "fixture".into()
            )))
            .is_err()
        );
        assert!(
            process_with(&[1.0], RATE, |_| Ok(WhisperLogMel {
                values: vec![0.0; 2],
                frames: FRAMES
            }))
            .is_err()
        );
        assert!(
            process_with(&[1.0], RATE, |_| Ok(WhisperLogMel {
                values: vec![f32::NAN; MEL_BINS * FRAMES],
                frames: FRAMES
            }))
            .is_err()
        );
    }
}

/// Bounded CPU stem for the Qwen2-Audio encoder.
#[cfg(feature = "qwen2-audio-stem")]
pub mod stem;

/// Bounded synthetic CPU encoder layer for Qwen2-Audio.
#[cfg(feature = "qwen2-audio-stem")]
pub mod layer;
