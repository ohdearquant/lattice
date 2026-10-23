use super::{GenerateConfig, sample_from_candidates};
use crate::sampling::Candidate;

#[test]
fn compact_metal_candidate_path_applies_min_p() {
    let candidates = [
        Candidate {
            token_id: 7,
            logit: 0.0,
        },
        Candidate {
            token_id: 8,
            logit: 0.49_f32.ln(),
        },
    ];
    let disabled = GenerateConfig {
        min_p: 0.0,
        temperature: 1.0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let enabled = GenerateConfig {
        min_p: 0.5,
        ..disabled.clone()
    };
    let mut saw_tail_without_min_p = false;

    for seed in [0x1234_5678_9abc_def0, 0xabad_1dea_dead_beef] {
        let mut control_rng = seed;
        saw_tail_without_min_p |=
            sample_from_candidates(&candidates, &disabled, &[], &mut control_rng) == 8;

        let mut filtered_rng = seed;
        assert_eq!(
            sample_from_candidates(&candidates, &enabled, &[], &mut filtered_rng),
            7,
            "compact Metal candidates below min_p * max_probability must be removed"
        );
    }

    assert!(
        saw_tail_without_min_p,
        "control seeds must exercise the tail candidate or the min-p assertion is vacuous"
    );
}
