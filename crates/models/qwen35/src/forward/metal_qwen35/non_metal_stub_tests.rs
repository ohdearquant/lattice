    use super::MetalQwen35State;
    use crate::generation::GenerateConfig;
    use lattice_inference_core::InferenceError;
    use lattice_tokenizer::bpe::BpeTokenizer;
    use std::collections::HashMap;

    fn tiny_tokenizer() -> BpeTokenizer {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
            vocab.insert((*c).to_string(), i as u32);
        }
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap()
    }

    #[test]
    fn non_metal_stub_try_forward_prefill_returns_capability_error() {
        let mut state = MetalQwen35State;
        let result = state.try_forward_prefill(&[1, 2]);
        match result {
            Err(InferenceError::Inference(message)) => {
                assert!(message.contains("metal-gpu"), "{message}");
            }
            other => panic!("expected Metal capability error, got {other:?}"),
        }
    }

    #[test]
    fn non_metal_stub_score_option_letters_returns_capability_error() {
        let mut state = MetalQwen35State;
        match state.score_option_letters(&[1, 2], &[3, 4]) {
            Err(InferenceError::Inference(message)) => {
                assert!(message.contains("metal-gpu"), "{message}");
            }
            other => panic!("expected Metal capability error, got {other:?}"),
        }
    }

    #[test]
    fn non_metal_stub_generate_rejects_empty_prompt() {
        let tokenizer = tiny_tokenizer();
        let gen_cfg = GenerateConfig::default();
        let mut state = MetalQwen35State;

        // Must return a typed Err (fail loud, not fail open) rather than a
        // successful-looking empty GenerateOutput for an empty prompt.
        let result = state.generate("", &tokenizer, &gen_cfg);
        match result {
            Err(InferenceError::Inference(msg)) => {
                assert!(
                    msg.contains("metal-gpu"),
                    "expected capability error mentioning metal-gpu, got: {msg}"
                );
            }
            other => panic!("expected Err(InferenceError::Inference(..)), got: {other:?}"),
        }
    }

    /// The stub errors for every prompt, not only the empty one --
    /// directly-constructing it (bypassing the fallible `new`/`from_q4_dir`)
    /// must never yield a successful (if vacuous) completion.
    #[test]
    fn non_metal_stub_generate_rejects_nonempty_prompt() {
        let tokenizer = tiny_tokenizer();
        let gen_cfg = GenerateConfig::default();
        let mut state = MetalQwen35State;

        let result = state.generate("hello", &tokenizer, &gen_cfg);
        match result {
            Err(InferenceError::Inference(msg)) => {
                assert!(
                    msg.contains("metal-gpu"),
                    "expected capability error mentioning metal-gpu, got: {msg}"
                );
            }
            other => panic!("expected Err(InferenceError::Inference(..)), got: {other:?}"),
        }
    }

    #[test]
    fn non_metal_stub_speculative_generation_returns_capability_error() {
        let mut state = MetalQwen35State;

        let result = state.generate_with_speculation(&[0], 1, u32::MAX, 5, 4);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(ref message))
                if message
                    == "Metal GPU not available (requires macOS + metal-gpu feature)"
        ));
    }
