use super::*;

fn limits() -> MergeLimits {
    MergeLimits {
        audio_token_id: 9,
        pad_token_id: 0,
        vocab_size: 16,
        hidden_size: 2,
        context_limit: 12,
        generation_budget: 2,
    }
}

#[test]
fn middle_marker_preserves_text_audio_order_and_legacy_ids() {
    let ids = [1, 9, 2, 3];
    let plan = SingleAudioPlan::new(&ids, &[1; 4], 2, limits()).unwrap();
    assert_eq!(plan.final_input_ids(), &[1, 0, 0, 2, 3]);
    assert_eq!(
        plan.sources(),
        &[
            RowSource::Text(0),
            RowSource::Audio(0),
            RowSource::Audio(1),
            RowSource::Text(2),
            RowSource::Text(3)
        ]
    );
    let text = [10., 11., 90., 91., 20., 21., 30., 31.];
    let audio = [40., 41., 50., 51.];
    let mut out = [-1.; 10];
    plan.splice(&ids, &text, &audio, &mut out).unwrap();
    assert_eq!(out, [10., 11., 40., 41., 50., 51., 20., 21., 30., 31.]);
}

#[test]
fn marker_at_edges_and_single_audio_row() {
    for ids in [[9, 1, 2], [1, 2, 9]] {
        let plan = SingleAudioPlan::new(&ids, &[1; 3], 1, limits()).unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!(
            plan.sources()
                .iter()
                .filter(|s| matches!(s, RowSource::Audio(0)))
                .count(),
            1
        );
        assert_eq!(
            plan.final_input_ids().iter().filter(|&&id| id == 0).count(),
            1
        );
    }
}

#[test]
fn longest_audio_fits_or_refuses_at_exact_context_boundary() {
    let mut l = limits();
    l.context_limit = 752;
    l.generation_budget = 1;
    let ids = [1, 9];
    let plan = SingleAudioPlan::new(&ids, &[1; 2], 750, l).unwrap();
    assert_eq!(plan.len(), 751);
    assert_eq!(plan.sources()[750], RowSource::Audio(749));
    l.generation_budget = 2;
    assert!(SingleAudioPlan::new(&ids, &[1; 2], 750, l).is_err());
    assert!(SingleAudioPlan::new(&ids, &[1; 2], 751, l).is_err());
    l.generation_budget = usize::MAX;
    assert!(SingleAudioPlan::new(&ids, &[1; 2], 1, l).is_err());
}

#[test]
fn rejects_missing_repeated_padded_or_invalid_marker_inputs() {
    for (ids, mask) in [
        (vec![], vec![]),
        (vec![1, 2], vec![1, 1]),
        (vec![9, 9], vec![1, 1]),
        (vec![9, 1], vec![0, 1]),
        (vec![9, 1], vec![1]),
        (vec![9, 0], vec![1, 1]),
        (vec![9, 16], vec![1, 1]),
    ] {
        assert!(SingleAudioPlan::new(&ids, &mask, 1, limits()).is_err());
    }
    assert!(SingleAudioPlan::new(&[9], &[1], 0, limits()).is_err());
    let mut l = limits();
    l.audio_token_id = l.pad_token_id;
    assert!(SingleAudioPlan::new(&[0], &[1], 1, l).is_err());
    l = limits();
    l.audio_token_id = 16;
    assert!(SingleAudioPlan::new(&[9], &[1], 1, l).is_err());
}

#[test]
fn splice_refusals_never_modify_output() {
    let ids = [1, 9, 2];
    let plan = SingleAudioPlan::new(&ids, &[1; 3], 2, limits()).unwrap();
    let text = [1., 2., 3., 4., 5., 6.];
    let audio = [7., 8., 9., 10.];
    let mut out = [42.; 8];
    assert!(plan.splice(&[1, 9, 3], &text, &audio, &mut out).is_err());
    assert_eq!(out, [42.; 8]);
    assert!(plan.splice(&ids, &text[..5], &audio, &mut out).is_err());
    assert_eq!(out, [42.; 8]);
    assert!(plan.splice(&ids, &text, &audio[..3], &mut out).is_err());
    assert_eq!(out, [42.; 8]);
    assert!(plan.splice(&ids, &text, &audio, &mut out[..7]).is_err());
    assert_eq!(out, [42.; 8]);
    let bad_audio = [7., f32::NAN, 9., 10.];
    assert!(plan.splice(&ids, &text, &bad_audio, &mut out).is_err());
    assert_eq!(out, [42.; 8]);
}

#[test]
fn invalid_resource_bounds_are_refused() {
    for (hidden, context, vocab) in [
        (0, 12, 16),
        (4097, 12, 16),
        (2, 0, 16),
        (2, 8193, 16),
        (2, 12, 0),
    ] {
        let mut l = limits();
        l.hidden_size = hidden;
        l.context_limit = context;
        l.vocab_size = vocab;
        assert!(SingleAudioPlan::new(&[9], &[1], 1, l).is_err());
    }
}

#[test]
fn validation_error_order_and_messages_are_preserved() {
    let message = |result: Result<SingleAudioPlan, InferenceError>| match result {
        Err(InferenceError::InvalidInput(message)) => message,
        other => panic!("expected invalid input, got {other:?}"),
    };
    let mut l = limits();
    l.hidden_size = 0;
    assert_eq!(
        message(SingleAudioPlan::new(&[], &[], 0, l)),
        "qwen2-audio merge: empty input or attention mask length mismatch"
    );
    assert_eq!(
        message(SingleAudioPlan::new(&[9], &[0], 0, l)),
        "qwen2-audio merge: only one unpadded sequence is supported"
    );
    assert_eq!(
        message(SingleAudioPlan::new(&[0], &[1], 0, l)),
        "qwen2-audio merge: audio, width, vocabulary or context bounds"
    );
    assert_eq!(
        message(SingleAudioPlan::new(&[0], &[1], 1, limits())),
        "qwen2-audio merge: invalid token identity or reserved pad token"
    );
    assert_eq!(
        message(SingleAudioPlan::new(&[1], &[1], 1, limits())),
        "qwen2-audio merge: expected exactly one audio marker"
    );
    l = limits();
    l.generation_budget = usize::MAX;
    assert_eq!(
        message(SingleAudioPlan::new(&[9], &[1], 1, l)),
        "qwen2-audio merge: expanded prompt and generation exceed context"
    );
}

/// Host-generated fixtures invoke the pinned HF implementation independently.
#[test]
#[ignore = "requires fixture generated by scripts/qwen2_audio_merge_hf.py and QWEN2_AUDIO_MERGE_FIXTURE"]
fn pinned_hf_single_clip_merge_differential() {
    let path = std::env::var("QWEN2_AUDIO_MERGE_FIXTURE").expect("fixture path required");
    let data = std::fs::read_to_string(path).unwrap();
    let root: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(root["transformers_version"], "4.45.2");
    let cases = root["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
        let input_mask: Vec<u8> =
            serde_json::from_value(case["input_attention_mask"].clone()).unwrap();
        let text: Vec<f32> = serde_json::from_value(case["text_embeddings"].clone()).unwrap();
        let audio: Vec<f32> = serde_json::from_value(case["audio_embeddings"].clone()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(case["final_embeddings"].clone()).unwrap();
        let expected_ids: Vec<u32> = serde_json::from_value(case["final_ids"].clone()).unwrap();
        let attention: Vec<u8> = serde_json::from_value(case["attention_mask"].clone()).unwrap();
        let positions: Vec<usize> = serde_json::from_value(case["position_ids"].clone()).unwrap();
        let mut l = limits();
        l.generation_budget = 0;
        l.context_limit = 8192;
        let rows = audio.len() / l.hidden_size;
        let plan = SingleAudioPlan::new(&ids, &input_mask, rows, l).unwrap();
        let mut out = vec![0.; plan.len() * l.hidden_size];
        plan.splice(&ids, &text, &audio, &mut out).unwrap();
        assert_eq!(out, expected);
        assert_eq!(plan.final_input_ids(), expected_ids.as_slice());
        assert_eq!(attention, vec![1; plan.len()]);
        assert_eq!(positions, (0..plan.len()).collect::<Vec<_>>());
    }
}
