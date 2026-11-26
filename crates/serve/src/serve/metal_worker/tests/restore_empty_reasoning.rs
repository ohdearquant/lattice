use super::*;

#[test]
fn restore_empty_reasoning_writes_the_generation_block_back_only_with_reasoning_off() {
    let tokenizer = thinking_tokenizer();
    let render = |reply: &str, enable_thinking: bool, restore_empty_reasoning: bool| {
        let messages = [
            ChatMessage::user("a"),
            ChatMessage::assistant(reply),
            ChatMessage::user("b"),
        ];
        let mut cfg = GenerateConfig::default();
        cfg.enable_thinking = enable_thinking;
        cfg.max_new_tokens = 9;
        let (prompt, _) = render_text_prompt_within_window(
            &tokenizer,
            &messages,
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            4096,
            &cfg,
            false,
            restore_empty_reasoning,
        )
        .expect("renders");
        template_text(&prompt)
    };
    let history = |assistant: &str, generation: &str| {
        format!(
            "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n\
             <|im_start|>user\nb<|im_end|>\n<|im_start|>assistant\n{generation}"
        )
    };
    let closed = "<think>\n\n</think>\n\n";
    let primed = "<think>\n";
    let echoed = "<think>\n\n</think>\n\nfor";
    let reasoned = "<think>\nbe\n</think>\n\nfor";

    // With reasoning off, a reply sent back as the plain answer gets the block its generation
    // prompt held, and a reply that echoes that block renders the same.
    assert_eq!(render("for", false, true), history(echoed, closed));
    assert_eq!(render(echoed, false, true), render("for", false, true));
    // Reasoning with text in it renders as the template renders it.
    assert_eq!(render(reasoned, false, true), history("for", closed));
    assert_eq!(
        render(reasoned, false, true),
        render(reasoned, false, false)
    );
    // A request with reasoning on renders as the template renders it.
    for reply in ["for", echoed, reasoned] {
        assert_eq!(
            render(reply, true, true),
            render(reply, true, false),
            "{reply:?}"
        );
    }
    assert_eq!(render("for", true, true), history("for", primed));
    // Unset, the history renders as the template renders it.
    assert_eq!(render("for", false, false), history("for", closed));
    assert_eq!(render(echoed, false, false), history("for", closed));

    // A reply with no text, which is what an assistant turn carrying only tool calls would be
    // here (the request contract refuses a non-empty `tool_calls` list and the `tool` role
    // before rendering, so only the empty text reaches this point), is written as the block
    // alone. Nothing was generated after that block, so the next turn's prompt begins with the
    // first turn's whole prompt and appends to it; unset, it parts from it at the block.
    let first = {
        let mut cfg = GenerateConfig::default();
        cfg.enable_thinking = false;
        cfg.max_new_tokens = 9;
        let (prompt, _) = render_text_prompt_within_window(
            &tokenizer,
            &[ChatMessage::user("a")],
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            4096,
            &cfg,
            false,
            true,
        )
        .expect("renders");
        template_text(&prompt)
    };
    assert_eq!(render("", false, true), history(closed, closed));
    assert!(render("", false, true).starts_with(&first), "{first:?}");
    assert_eq!(render("", false, false), history("", closed));
    assert!(!render("", false, false).starts_with(&first));
}

/// The append the setting allows is to the tokens actually cached. On a real checkpoint, with
/// reasoning off and the worker's own rendering and generation, the second turn's prompt ids
/// begin with the first turn's prompt ids followed, position for position, by the ids the first
/// turn generated, and the cache appends to them. Unset, the same conversation replays from the
/// first turn's start. Needs `LATTICE_PREFIX_ROUTE_MODEL_DIR` (a Q4 checkpoint) and optionally
/// `LATTICE_PREFIX_ROUTE_TOKENIZER_DIR`; with the first unset it prints one `SKIP` line and
/// passes.
#[test]
fn restore_empty_reasoning_appends_to_the_ids_the_first_turn_generated_on_a_real_checkpoint() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::kv_cache::{CrossTurnSlotId, PrefixReuseMode};
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::serving_support::metal_generate_streaming_with_prefix_cache_ids_traced;
    use std::path::PathBuf;

    let Some(model_dir) = std::env::var_os("LATTICE_PREFIX_ROUTE_MODEL_DIR").map(PathBuf::from)
    else {
        eprintln!(
            "SKIP restore empty reasoning real checkpoint: LATTICE_PREFIX_ROUTE_MODEL_DIR unset"
        );
        return;
    };
    let tokenizer_dir = std::env::var_os("LATTICE_PREFIX_ROUTE_TOKENIZER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.clone());
    let _gpu_guard = lattice_inference::measurement::gpu_test_lock();

    let tokenizer_path = tokenizer_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)
        .unwrap_or_else(|error| panic!("tokenizer {}: {error}", tokenizer_path.display()));
    let config = Qwen35Config::from_model_dir(&model_dir)
        .unwrap_or_else(|error| panic!("config.json in {}: {error}", model_dir.display()));
    const MAX_NEW_TOKENS: usize = 24;
    const CONTEXT: usize = 1024;
    let cfg = crate::serve::prepare::lattice_gen_cfg_with_thinking(
        MAX_NEW_TOKENS,
        0.0,
        1.0,
        Some(7),
        Vec::new(),
        None,
        None,
        false,
    );
    // The reply must come back as the ids generated. The template trims a reply, so one that
    // begins with white space re-tokenizes differently and replays; this question's reply on
    // the 0.8B Q4 checkpoint ends on its own and begins with a word.
    let q1 = "Translate 'cat' to French.";
    let q2 = "And 'dog'?";

    let run = |restore_empty_reasoning: bool| {
        let mut state =
            MetalQwen35State::from_q4_dir(&model_dir, &tokenizer_path, &config, CONTEXT)
                .unwrap_or_else(|error| panic!("from_q4_dir {}: {error}", model_dir.display()));
        let render = |messages: &[ChatMessage]| {
            render_text_prompt_within_window(
                &tokenizer,
                messages,
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                CONTEXT,
                &cfg,
                false,
                restore_empty_reasoning,
            )
            .expect("the conversation fits the window")
            .1
        };
        let first_ids = render(&[ChatMessage::user(q1)]);
        let (first, _) = metal_generate_streaming_with_prefix_cache_ids_traced(
            &mut state,
            CrossTurnSlotId::DEFAULT,
            first_ids.clone(),
            &tokenizer,
            &cfg,
            |_, _| true,
            || false,
        )
        .expect("first turn");
        let reply = first.output.text.clone();
        let second_ids = render(&[
            ChatMessage::user(q1),
            ChatMessage::assistant(reply.clone()),
            ChatMessage::user(q2),
        ]);
        let (second, _) = metal_generate_streaming_with_prefix_cache_ids_traced(
            &mut state,
            CrossTurnSlotId::DEFAULT,
            second_ids.clone(),
            &tokenizer,
            &cfg,
            |_, _| true,
            || false,
        )
        .expect("second turn");
        eprintln!(
            "restore empty reasoning real case={restore_empty_reasoning} reply={reply:?} \
             first_prompt={} generated={} second_mode={:?} reused={} prefetched={} prompt={}",
            first_ids.len(),
            first.output.token_ids.len(),
            second.cache.mode,
            second.cache.reused_tokens,
            second.cache.prefetched_tokens,
            second.cache.prompt_tokens,
        );
        let mut held = first_ids.clone();
        held.extend_from_slice(&first.output.token_ids);
        (first, held, second_ids, second)
    };

    let (first, held, second_ids, second) = run(true);
    assert!(
        first.output.stopped && !first.output.token_ids.is_empty(),
        "precondition: the first reply must end on its own before {MAX_NEW_TOKENS} tokens so it \
         comes back whole; got {:?}",
        first.output.text
    );
    assert_eq!(
        second_ids.get(..held.len()),
        Some(held.as_slice()),
        "the second prompt must begin with the first prompt and the generated ids, position for \
         position"
    );
    assert_eq!(second.cache.mode, PrefixReuseMode::ExactAppend);
    assert!(
        second.cache.reused_tokens >= held.len(),
        "the append must reuse at least the first prompt and the generated ids: {} < {}",
        second.cache.reused_tokens,
        held.len()
    );

    let (template_first, template_held, template_second_ids, template_second) = run(false);
    assert_eq!(
        template_first.output.token_ids, first.output.token_ids,
        "precondition: the setting renders history only, so the first greedy reply must match"
    );
    assert_ne!(
        template_second_ids.get(..template_held.len()),
        Some(template_held.as_slice()),
        "the published rendering drops the block, so the cached ids are not a prefix"
    );
    assert!(
        matches!(
            template_second.cache.mode,
            PrefixReuseMode::ReplayFromCheckpoint { .. }
        ),
        "unset, the second turn must replay; got {:?}",
        template_second.cache.mode
    );
    assert!(
        second.cache.prefetched_tokens < template_second.cache.prefetched_tokens,
        "appending must prefill fewer tokens than the replay: {} vs {}",
        second.cache.prefetched_tokens,
        template_second.cache.prefetched_tokens
    );
}
