use super::*;
use crate::model::qwen35::test_support::{
    tiny_zero_model, tiny_zero_model_with_context, tiny_zero_model_with_tokenizer,
};
use crate::stop_reason::StopReason;
use lattice_arch::decoder::driver;

const THINK_TOKENIZER: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,
  "added_tokens":[{"id":7,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#;

// Token 0 decodes to a line ending, and a zero-weight model always picks token 0.
const NEWLINE_THINK_TOKENIZER: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,
  "added_tokens":[{"id":7,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"Ċ":0,"<unk>":1,"a":2,"b":3,"c":4,"d":5,"e":6},"merges":[]}
}"#;

fn greedy(max_new_tokens: usize) -> GenerateConfig {
    let mut config = GenerateConfig::default();
    config.max_new_tokens = max_new_tokens;
    config.temperature = 0.0;
    config.stop_token_ids = vec![];
    config.enable_thinking = false;
    config
}

fn budgeted(max_new_tokens: usize, reasoning_budget: usize) -> GenerateConfig {
    let mut config = greedy(max_new_tokens);
    config.enable_thinking = true;
    config.reasoning_budget = Some(reasoning_budget);
    config
}

type Run = (Vec<u32>, String, StopReason);

fn run_generate(
    model: &Qwen35Model,
    prompt: &str,
    config: &GenerateConfig,
) -> Result<Run, InferenceError> {
    model.generate(prompt, config).map(|output| {
        (
            output.token_ids,
            output.text,
            output.stop_reason.expect("generate reports a stop reason"),
        )
    })
}

/// The admitted path a consumer of the family contract runs: admit, open a session, drive it
/// with the shared driver, and decode with the model's text decoder.
fn run_admitted(
    decoder: &mut Qwen35CpuDecoder,
    prompt: &str,
    config: &GenerateConfig,
) -> Result<Run, InferenceError> {
    let request = decoder.admit(prompt, config.clone())?;
    let text = decoder.text();
    let mut stream = text.stream();
    let mut session = decoder.session(&request)?;
    let mut full = String::new();
    let mut offsets = Vec::new();
    let never_cancel = || false;
    let result = driver::run_into(
        session.as_mut(),
        request.config(),
        request.think_close_id(),
        request.reasoning_newline_boundary().cloned(),
        request.prompt_ids(),
        request.eos_token_id(),
        false,
        &never_cancel,
        |_| {},
        |id, out| stream.push_into(id, out),
        &mut full,
        &mut offsets,
        |_, _| true,
        || {},
        String::new,
    )?;
    full.push_str(&stream.finish());
    Ok((result.generated_ids, full, result.stop_reason))
}

fn assert_same_run(decoder: &mut Qwen35CpuDecoder, prompt: &str, config: &GenerateConfig) {
    let expected = run_generate(decoder.model(), prompt, config);
    let admitted = run_admitted(decoder, prompt, config);
    match (&expected, &admitted) {
        (Ok(expected), Ok(admitted)) => assert_eq!(admitted, expected, "prompt {prompt:?}"),
        (Err(expected), Err(admitted)) => {
            assert_eq!(
                admitted.to_string(),
                expected.to_string(),
                "prompt {prompt:?}"
            )
        }
        _ => {
            panic!("prompt {prompt:?}: generate gave {expected:?}, the admitted path {admitted:?}")
        }
    }
}

#[test]
fn admit_tokenizes_the_prompt_as_generate_does() {
    let mut decoder = Qwen35CpuDecoder::new(tiny_zero_model());
    for prompt in ["a", "abcde", "edcba"] {
        let request = decoder.admit(prompt, greedy(3)).expect("admitted");
        let input = decoder.model().tokenizer().tokenize(prompt);
        assert_eq!(request.prompt_ids(), &input.input_ids[..input.real_length]);
        let output = decoder
            .model()
            .generate(prompt, &greedy(3))
            .expect("generated");
        assert_eq!(
            output.prompt_tokens,
            request.prompt_ids().len(),
            "prompt {prompt:?}"
        );
        assert_same_run(&mut decoder, prompt, &greedy(3));
    }
    assert_eq!(
        decoder
            .admit("edcba", greedy(3))
            .expect("admitted")
            .prompt_ids(),
        &[5, 4, 3, 2, 1]
    );
    // Tokenization truncates at the tokenizer's cap on both paths, and both then refuse the
    // truncated prompt against the context window with the same error.
    let cap = decoder.model().tokenizer().max_seq_len();
    assert!(cap > decoder.model().max_context());
    let long = "ab".repeat(cap);
    assert_eq!(decoder.model().tokenizer().tokenize(&long).real_length, cap);
    assert!(matches!(
        decoder.admit(&long, greedy(3)),
        Err(InferenceError::Inference(message)) if message.contains(&format!("prompt ({cap} tokens)"))
    ));
    assert_same_run(&mut decoder, &long, &greedy(3));
    assert!(matches!(
        decoder.admit("", greedy(3)),
        Err(InferenceError::Inference(_))
    ));
    assert_same_run(&mut decoder, "", &greedy(3));
}

#[test]
fn admit_applies_the_context_budget_generate_applies() {
    let mut decoder = Qwen35CpuDecoder::new(tiny_zero_model_with_context(8));
    assert_eq!(decoder.model().max_context(), 8);
    assert!(decoder.admit("abc", greedy(5)).is_ok());
    assert!(matches!(
        decoder.admit("abc", greedy(6)),
        Err(InferenceError::Inference(_))
    ));
    for config in [greedy(5), greedy(6), budgeted(1, 3), budgeted(1, 4)] {
        assert_same_run(&mut decoder, "abc", &config);
    }
    // A zero cap generates nothing, so neither path checks the budget.
    let request = decoder
        .admit("abcdeabcde", greedy(0))
        .expect("a zero cap is admitted");
    assert_eq!(request.think_close_id(), None);
    assert_same_run(&mut decoder, "abcdeabcde", &greedy(0));
}

#[test]
fn admit_resolves_the_reasoning_close_token_generate_resolves() {
    let mut decoder = Qwen35CpuDecoder::new(tiny_zero_model_with_tokenizer(THINK_TOKENIZER));
    let request = decoder.admit("a", budgeted(8, 2)).expect("admitted");
    assert_eq!(request.think_close_id(), Some(7));
    assert_same_run(&mut decoder, "a", &budgeted(8, 2));

    // No budget, or thinking off: no close token is resolved.
    assert_eq!(
        decoder
            .admit("a", greedy(8))
            .expect("admitted")
            .think_close_id(),
        None
    );
    let mut off = budgeted(8, 2);
    off.enable_thinking = false;
    assert_eq!(
        decoder
            .admit("a", off.clone())
            .expect("admitted")
            .think_close_id(),
        None
    );
    assert_same_run(&mut decoder, "a", &off);

    // A tokenizer without the marker refuses a budget on both paths.
    let mut plain = Qwen35CpuDecoder::new(tiny_zero_model());
    assert!(matches!(
        plain.admit("a", budgeted(8, 2)),
        Err(InferenceError::InvalidInput(_))
    ));
    assert_same_run(&mut plain, "a", &budgeted(8, 2));
}

#[test]
fn admit_carries_the_line_ending_table_generate_uses() {
    let mut decoder =
        Qwen35CpuDecoder::new(tiny_zero_model_with_tokenizer(NEWLINE_THINK_TOKENIZER));
    let request = decoder.admit("a", budgeted(12, 2)).expect("admitted");
    let table = request
        .reasoning_newline_boundary()
        .expect("a budgeted request carries the table");
    assert_eq!(*table, newline_boundary_table(decoder.model().tokenizer()));
    assert!(
        table[0] && !table[2],
        "token 0 ends a line, token 2 does not"
    );
    assert_same_run(&mut decoder, "a", &budgeted(12, 2));
    // Token 0 ends a line and the zero-weight model always picks it, so the budget closes the
    // reasoning block with the admitted close token at the first boundary past the budget.
    let (ids, _, _) = run_admitted(&mut decoder, "a", &budgeted(12, 2)).expect("ran");
    assert!(
        ids.contains(&7),
        "the budget forces the close token: {ids:?}"
    );
    assert_eq!(
        decoder
            .admit("a", greedy(12))
            .expect("admitted")
            .reasoning_newline_boundary(),
        None
    );
}

#[test]
fn admit_ends_at_the_end_of_sequence_id_generate_uses() {
    let mut model = tiny_zero_model();
    model.config.eos_token_id = 0;
    let mut decoder = Qwen35CpuDecoder::new(model);
    assert_eq!(
        decoder
            .admit("a", greedy(4))
            .expect("admitted")
            .eos_token_id(),
        0
    );
    assert_same_run(&mut decoder, "a", &greedy(4));
    let (_, _, stop) = run_admitted(&mut decoder, "a", &greedy(4)).expect("ran");
    assert_eq!(stop, StopReason::Eos);

    let mut decoder = Qwen35CpuDecoder::new(tiny_zero_model());
    assert_eq!(
        decoder
            .admit("a", greedy(4))
            .expect("admitted")
            .eos_token_id(),
        96
    );
    assert_same_run(&mut decoder, "a", &greedy(4));
    let (ids, _, stop) = run_admitted(&mut decoder, "a", &greedy(4)).expect("ran");
    assert_eq!((ids.len(), stop), (4, StopReason::Length));
}

fn refusal(decoder: &mut Qwen35CpuDecoder, request: &SessionRequest) -> InferenceError {
    match decoder.session(request) {
        Ok(_) => panic!("the session opened"),
        Err(error) => error,
    }
}

#[test]
fn session_refuses_a_request_the_model_does_not_serve() {
    let mut decoder = Qwen35CpuDecoder::new(tiny_zero_model_with_tokenizer(THINK_TOKENIZER));
    let admitted = decoder.admit("ab", budgeted(4, 2)).expect("admitted");
    let close = admitted.think_close_id().expect("a close token");
    let table = admitted
        .reasoning_newline_boundary()
        .expect("a table")
        .clone();
    assert!(decoder.session(&admitted).is_ok());

    let invalid = [
        SessionRequest::new(vec![1, 97], greedy(4), 96),
        SessionRequest::new(vec![1], greedy(4), 95),
        SessionRequest::new(vec![1, 2], budgeted(4, 2), 96),
        SessionRequest::new(vec![1, 2], greedy(4), 96).with_reasoning_close(close, table.clone()),
        SessionRequest::new(vec![1, 2], budgeted(4, 2), 96).with_reasoning_close(5, table.clone()),
        SessionRequest::new(vec![1, 2], budgeted(4, 2), 96)
            .with_reasoning_close(close, Arc::from(vec![true; table.len()])),
    ];
    for request in &invalid {
        let error = refusal(&mut decoder, request);
        assert!(matches!(error, InferenceError::InvalidInput(_)), "{error}");
    }
    for request in [
        SessionRequest::new(vec![], greedy(4), 96),
        SessionRequest::new(vec![1, 2], greedy(1023), 96),
    ] {
        let error = refusal(&mut decoder, &request);
        assert!(matches!(error, InferenceError::Inference(_)), "{error}");
    }

    // A zero cap generates nothing, so the session does not check the close token.
    assert!(
        decoder
            .session(&SessionRequest::new(vec![1, 2], budgeted(0, 2), 96))
            .is_ok()
    );
}

#[test]
fn the_text_decoder_decodes_and_streams_the_models_tokens() {
    let mut decoder = Qwen35CpuDecoder::new(tiny_zero_model());
    let text = decoder.text();
    assert_eq!(text.decode(&[1, 2, 5, 3]), "abec");
    let mut stream = text.stream();
    let mut pieces = Vec::new();
    for id in [1, 5, 2] {
        let mut out = String::new();
        stream.push_into(id, &mut out);
        pieces.push(out);
    }
    assert_eq!(pieces, ["a", "e", "b"]);
    assert_eq!(stream.finish(), "");
    // The decoder does not borrow the owner: a session can open while it is alive.
    let request = decoder.admit("a", greedy(1)).expect("admitted");
    assert!(decoder.session(&request).is_ok());
    assert_eq!(text.decode(&[5]), "e");
}

#[test]
fn the_registry_resolves_each_family_by_kind() {
    let registry = registry().expect("the crate's table is valid");
    let name = |key, kind| registry.resolve(key, kind).map(FamilyDescriptor::name);
    assert_eq!(
        name(ResolveKey::ModelType("qwen3_5_moe"), ModelKind::Decoder),
        Ok("qwen35")
    );
    #[cfg(feature = "bert")]
    assert_eq!(
        name(ResolveKey::ModelType("bert"), ModelKind::Encoder),
        Ok("bert")
    );
    #[cfg(feature = "embeddinggemma2")]
    assert_eq!(
        name(ResolveKey::Family("embeddinggemma2"), ModelKind::Encoder),
        Ok("embeddinggemma2")
    );
    #[cfg(feature = "bert")]
    assert_eq!(
        name(ResolveKey::ModelType("bert"), ModelKind::Decoder),
        Err(RegistryError::WrongKind {
            requested: ModelKind::Decoder,
            families: vec!["bert"],
        })
    );
    assert_eq!(
        name(ResolveKey::ModelType("qwen3"), ModelKind::Encoder),
        Err(RegistryError::UnregisteredModelType {
            model_type: "qwen3".into()
        })
    );
    let metal = LoadRequest::new("/nonexistent").with_backend(Backend::Metal);
    assert!(matches!(
        QWEN35.load_decoder(&metal).map(|_| ()),
        Err(InferenceError::UnsupportedModel(message)) if message == "family qwen35 does not offer the metal backend"
    ));
    assert!(
        QWEN35
            .load_decoder(&LoadRequest::new("/nonexistent"))
            .is_err()
    );
    #[cfg(feature = "bert")]
    assert!(
        BERT.load_encoder(&LoadRequest::new("/nonexistent"))
            .is_err()
    );
}

#[cfg(not(feature = "embeddinggemma2"))]
#[test]
fn registry_omits_disabled_embeddinggemma2() {
    assert!(
        registry()
            .unwrap()
            .resolve(ResolveKey::Family("embeddinggemma2"), ModelKind::Encoder)
            .is_err()
    );
}

#[test]
#[cfg(not(feature = "bert"))]
fn registry_does_not_register_a_disabled_bert_family() {
    let registry = registry().expect("the enabled family table is valid");
    assert!(matches!(
        registry.resolve(ResolveKey::Family("bert"), ModelKind::Encoder),
        Err(RegistryError::UnavailableFamily { .. })
    ));
}
