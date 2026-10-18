use super::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

struct BlockingThreadLocalDrop {
    entered: std::sync::mpsc::SyncSender<()>,
    release: std::sync::mpsc::Receiver<()>,
    finished: std::sync::mpsc::SyncSender<()>,
}

impl Drop for BlockingThreadLocalDrop {
    fn drop(&mut self) {
        let _ = self.entered.send(());
        let _ = self.release.recv();
        let _ = self.finished.send(());
    }
}

thread_local! {
    static BLOCKING_THREAD_LOCAL_DROP: RefCell<Option<BlockingThreadLocalDrop>> =
        const { RefCell::new(None) };
}

fn test_owner(
    join_handle: std::thread::JoinHandle<()>,
    drop_timeout: Duration,
) -> MetalWorkerOwner {
    MetalWorkerOwner::from_handle_with_timeout(join_handle, drop_timeout)
}

#[test]
fn measurement_worker_guard_stays_on_the_worker_after_owner_timeout() {
    let lock = Arc::new(Mutex::new(()));
    let worker_lock = Arc::clone(&lock);
    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
    let caller = std::thread::current().id();
    let handle = std::thread::spawn(move || {
        run_guarded_worker(
            || {
                assert_ne!(std::thread::current().id(), caller);
                worker_lock.lock().unwrap()
            },
            || {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            },
        );
        done_tx.send(()).unwrap();
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let owner = test_owner(handle, Duration::from_millis(20));
    assert_eq!(
        owner._inner.wait_for_exit(Duration::from_millis(20)),
        WorkerShutdown::TimedOut,
    );
    assert!(matches!(
        lock.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    release_tx.send(()).unwrap();
    done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(lock.try_lock().is_ok());
}

#[test]
fn measurement_worker_guard_outlives_work_capture_destruction_and_unwind() {
    struct WorkDrop(Arc<AtomicBool>);
    impl Drop for WorkDrop {
        fn drop(&mut self) {
            assert!(self.0.load(Ordering::SeqCst));
        }
    }
    struct GuardDrop(Arc<AtomicBool>);
    impl Drop for GuardDrop {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    for panic_in_work in [false, true] {
        let held = Arc::new(AtomicBool::new(false));
        let work_capture = WorkDrop(Arc::clone(&held));
        let result = std::panic::catch_unwind(|| {
            run_guarded_worker(
                || {
                    held.store(true, Ordering::SeqCst);
                    GuardDrop(Arc::clone(&held))
                },
                move || {
                    let _capture = work_capture;
                    assert!(!panic_in_work, "injected worker failure");
                },
            );
        });
        assert_eq!(result.is_err(), panic_in_work);
        assert!(!held.load(Ordering::SeqCst));
    }
}

#[test]
fn measurement_worker_guard_releases_after_a_real_factory_load_error() {
    struct GuardDrop(Arc<AtomicBool>);
    impl Drop for GuardDrop {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let held = Arc::new(AtomicBool::new(false));
    let guard_state = Arc::clone(&held);
    let loader_state = Arc::clone(&held);
    let caller = std::thread::current().id();
    let factory = ServingFactory::qwen_metal(
        move || {
            assert!(loader_state.load(Ordering::SeqCst));
            Err("injected load failure".into())
        },
        VisionRuntime::unsupported(),
    );
    let result = MetalWorker::spawn_with_guard(factory, 1, ResidencyLimits::default(), move || {
        assert_ne!(std::thread::current().id(), caller);
        guard_state.store(true, Ordering::SeqCst);
        GuardDrop(guard_state)
    });
    assert!(
        matches!(result, Err(StartupError::Load(ref message)) if message == "injected load failure")
    );
    assert!(!held.load(Ordering::SeqCst));
}

fn tiny_tokenizer() -> BpeTokenizer {
    BpeTokenizer::from_vocab_and_merges(
        HashMap::from([
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("e".to_string(), 2),
            ("f".to_string(), 3),
            ("o".to_string(), 4),
            ("r".to_string(), 5),
        ]),
        Vec::new(),
    )
    .expect("tiny tokenizer must construct")
}

fn tiny_vision_config(out_hidden_size: usize) -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 2,
        patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size,
        temporal_patch_size: 1,
        num_position_embeddings: 16,
        in_channels: 3,
        deepstack_visual_indexes: Vec::new(),
        intermediate_size: None,
    }
}

fn thinking_tokenizer() -> BpeTokenizer {
    BpeTokenizer::from_vocab_and_merges(
        "abeforthink"
            .chars()
            .enumerate()
            .map(|(id, ch)| (ch.to_string(), u32::try_from(id).expect("tiny vocabulary")))
            .collect(),
        Vec::new(),
    )
    .expect("thinking tokenizer")
}

#[test]
fn text_worker_renders_omitted_thinking_as_a_closed_block() {
    let messages = [ChatMessage::user("hello")];
    let tokenizer = thinking_tokenizer();
    let cfg = GenerateConfig {
        enable_thinking: false,
        reasoning_budget: None,
        max_new_tokens: 9,
        ..GenerateConfig::default()
    };
    let (prompt, _) = render_text_prompt_within_window(
        &tokenizer,
        &messages,
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        &cfg,
        false,
    )
    .expect("omitted thinking is off");
    assert_eq!(
        prompt,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn text_worker_renders_positive_request_budget_with_the_thinking_prime() {
    let messages = [ChatMessage::user("hello")];
    let tokenizer = thinking_tokenizer();
    let cfg = GenerateConfig {
        enable_thinking: true,
        reasoning_budget: Some(16),
        max_new_tokens: 9,
        ..GenerateConfig::default()
    };
    let (prompt, _) = render_text_prompt_within_window(
        &tokenizer,
        &messages,
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        &cfg,
        false,
    )
    .expect("positive request budget enables thinking");
    assert_eq!(
        prompt,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn text_worker_renders_explicit_thinking_true_with_the_thinking_prime() {
    let messages = [ChatMessage::user("hello")];
    let tokenizer = thinking_tokenizer();
    let cfg = GenerateConfig {
        enable_thinking: true,
        max_new_tokens: 9,
        ..GenerateConfig::default()
    };
    let (prompt, _) = render_text_prompt_within_window(
        &tokenizer,
        &messages,
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        &cfg,
        true,
    )
    .expect("explicit thinking prime");
    assert_eq!(
        prompt,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn text_worker_renders_explicit_thinking_false_as_a_closed_block() {
    let messages = [ChatMessage::user("hello")];
    let tokenizer = thinking_tokenizer();
    let cfg = GenerateConfig {
        enable_thinking: false,
        max_new_tokens: 9,
        ..GenerateConfig::default()
    };
    let (prompt, _) = render_text_prompt_within_window(
        &tokenizer,
        &messages,
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        &cfg,
        false,
    )
    .expect("explicit thinking off");
    assert_eq!(
        prompt,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn vision_worker_renders_omitted_thinking_as_a_closed_block() {
    let messages = [ChatMessage::user_with_image(
        "beforeafter",
        vec![1],
        "before".len(),
    )];
    let tokenizer = thinking_tokenizer();
    let (before, after) =
        build_vision_prompt_text(&messages, 0, false, false).expect("vision text");
    assert_eq!(before, "<|im_start|>user\nbefore");
    assert_eq!(
        after,
        "after<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let ids = build_vision_prompt_ids(&messages, 0, &tokenizer, 90, 91, 92, 3, false, false)
        .expect("closed vision prompt");
    let expected = tokenizer.tokenize_fragments_with_inserted_ids(
        "<|im_start|>user\nbefore",
        &[90, 92, 92, 92, 91],
        "after<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    assert_eq!(ids, expected);
}

#[test]
fn vision_worker_renders_enabled_thinking_with_the_thinking_prime() {
    let messages = [ChatMessage::user_with_image(
        "beforeafter",
        vec![1],
        "before".len(),
    )];
    let (_, after) = build_vision_prompt_text(&messages, 0, true, false).expect("vision text");
    assert_eq!(after, "after<|im_end|>\n<|im_start|>assistant\n<think>\n");
}

#[test]
fn vision_worker_renders_explicit_thinking_true_with_the_thinking_prime() {
    let messages = [ChatMessage::user_with_image(
        "beforeafter",
        vec![1],
        "before".len(),
    )];
    let (_, after) = build_vision_prompt_text(&messages, 0, true, true).expect("vision text");
    assert_eq!(after, "after<|im_end|>\n<|im_start|>assistant\n<think>\n");
}

#[test]
fn vision_worker_renders_explicit_thinking_false_as_a_closed_block() {
    let messages = [ChatMessage::user_with_image(
        "beforeafter",
        vec![1],
        "before".len(),
    )];
    let (_, after) = build_vision_prompt_text(&messages, 0, false, false).expect("vision text");
    assert_eq!(
        after,
        "after<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn vision_thinking_mode_reaches_exact_token_window_preflight() {
    let mut runtime = VisionRuntime::unsupported();
    let config = vision_build_config();
    let tokenizer = thinking_tokenizer();
    let messages = [ChatMessage::user_with_image(
        "beforeafter",
        make_test_png(8, 8),
        "before".len(),
    )];
    let mut counts = Vec::new();
    for enabled in [true, false] {
        let error = build_vision_request(
            &mut runtime,
            &config,
            &tokenizer,
            &messages,
            0,
            enabled,
            false,
            &mut || false,
            |count| {
                counts.push(count);
                Err(ApiError::BadRequest {
                    message: "test window refusal before lazy weights".to_owned(),
                    code: "context_length_exceeded",
                })
            },
        )
        .expect_err("test refuses before checkpoint loading or Metal dispatch");
        assert!(matches!(
            error,
            WorkerFailure::Rejected(ApiError::BadRequest {
                code: "context_length_exceeded",
                ..
            })
        ));
    }
    assert_eq!(counts.len(), 2);
    assert!(
        counts[1] > counts[0],
        "closed tail must be counted: {counts:?}"
    );
}

fn make_test_png(width: u32, height: u32) -> Vec<u8> {
    let mut image = image::RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y) % 256) as u8;
            image.put_pixel(x, y, image::Rgb([value, value, value]));
        }
    }
    let mut bytes = Vec::new();
    image
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("test PNG encode");
    bytes
}

#[test]
fn image_job_classification_enforces_role_and_multiplicity() {
    assert_eq!(
        classify_job(&[ChatMessage::user("hello")]).expect("text job"),
        JobRoute::Text
    );
    let image = ChatMessage::user_with_image("beforeafter", vec![1, 2, 3], 6);
    assert_eq!(
        classify_job(&[ChatMessage::system("policy"), image.clone()]).expect("one image job"),
        JobRoute::Vision { message_index: 1 }
    );
    let err = classify_job(&[image.clone(), image]).expect_err("two images must fail");
    assert!(matches!(
        err,
        WorkerFailure::Rejected(ApiError::BadRequest {
            code: "multiple_images_unsupported",
            ..
        })
    ));
    for role in [
        crate::forward::metal_qwen35::ChatRole::System,
        crate::forward::metal_qwen35::ChatRole::Assistant,
    ] {
        let mut non_user_image = ChatMessage::user_with_image("beforeafter", vec![1, 2, 3], 6);
        non_user_image.role = role;
        let err = classify_job(&[non_user_image]).expect_err("non-user image role must fail");
        assert!(matches!(
            err,
            WorkerFailure::Rejected(ApiError::BadRequest {
                code: "invalid_image_role",
                ..
            })
        ));
    }
}

#[test]
fn vision_prompt_splices_tokens_at_original_content_part_position() {
    let tokenizer = tiny_tokenizer();
    let messages = vec![
        ChatMessage::system("policy"),
        ChatMessage::user_with_image("beforeafter", vec![1], "before".len()),
        ChatMessage::assistant("prior"),
    ];
    let actual = build_vision_prompt_ids(&messages, 1, &tokenizer, 90, 91, 92, 3, true, false)
        .expect("vision prompt");

    let before = "<|im_start|>system\npolicy<|im_end|>\n<|im_start|>user\nbefore";
    let after =
        "after<|im_end|>\n<|im_start|>assistant\nprior<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let mut expected = tokenize_text(&tokenizer, before);
    expected.extend([90, 92, 92, 92, 91]);
    expected.extend(tokenize_text(&tokenizer, after));
    assert_eq!(actual, expected);
}

#[test]
fn vision_prompt_text_matches_plain_template_around_image_splice() {
    let messages = vec![
        ChatMessage::system("policy"),
        ChatMessage::user_with_image("beforeafter", vec![1], "before".len()),
        ChatMessage::assistant("prior"),
    ];
    let (before, after) =
        build_vision_prompt_text(&messages, 1, false, false).expect("vision text");
    let plain = format_chat_template(&messages);
    assert_eq!(
        before.clone()
            + after
                .strip_suffix("<think>\n\n</think>\n\n")
                .expect("closed suffix"),
        plain,
        "vision splice fragments must reconstruct exactly the canonical text-only template"
    );
}

#[test]
fn vision_prompt_tokenizes_the_complete_sequence_before_window_validation() {
    let capped = tiny_tokenizer().with_max_seq_len(4);
    let unbounded = capped.with_max_seq_len(usize::MAX);
    let before = "before".repeat(64);
    let messages = vec![ChatMessage::user_with_image(
        format!("{before}after"),
        vec![1],
        before.len(),
    )];
    let actual = build_vision_prompt_ids(&messages, 0, &capped, 90, 91, 92, 3, true, false)
        .expect("capped tokenizer must not truncate prompt fragments");
    let expected = build_vision_prompt_ids(&messages, 0, &unbounded, 90, 91, 92, 3, true, false)
        .expect("unbounded tokenizer");
    assert_eq!(actual, expected);
    assert!(actual.len() > 4);
}

fn vision_build_config() -> Qwen35Config {
    let mut config = Qwen35Config::qwen35_0_8b();
    config.vision_config = Some(tiny_vision_config(config.hidden_size));
    config.image_token_id = Some(90);
    config.vision_start_token_id = Some(91);
    config.vision_end_token_id = Some(92);
    config
}

#[test]
fn vision_request_build_cancels_before_image_preprocessing() {
    let mut runtime = VisionRuntime::unsupported();
    let config = vision_build_config();
    let messages = vec![ChatMessage::user_with_image(
        "beforeafter",
        b"not an image".to_vec(),
        "before".len(),
    )];
    let mut polls = 0;
    let result = build_vision_request(
        &mut runtime,
        &config,
        &tiny_tokenizer(),
        &messages,
        0,
        true,
        false,
        &mut || {
            polls += 1;
            true
        },
        |_| panic!("window preflight must not run after cancellation"),
    )
    .expect("cancellation is not a worker failure");
    assert!(matches!(result, VisionRequestBuild::Cancelled));
    assert_eq!(polls, 1);
}

#[test]
fn vision_request_build_cancels_after_preprocess_and_prompt_before_window_check() {
    let mut runtime = VisionRuntime::unsupported();
    let config = vision_build_config();
    let messages = vec![ChatMessage::user_with_image(
        "beforeafter",
        make_test_png(8, 8),
        "before".len(),
    )];
    let mut polls = 0;
    let mut window_checked = false;
    let result = build_vision_request(
        &mut runtime,
        &config,
        &tiny_tokenizer(),
        &messages,
        0,
        true,
        false,
        &mut || {
            polls += 1;
            polls == 3
        },
        |_| {
            window_checked = true;
            Ok(())
        },
    )
    .expect("cancellation is not a worker failure");
    assert!(matches!(result, VisionRequestBuild::Cancelled));
    assert_eq!(polls, 3);
    assert!(!window_checked);
}

#[test]
fn vision_request_build_cancels_after_window_check_before_lazy_load() {
    let mut runtime = VisionRuntime::unsupported();
    let config = vision_build_config();
    let messages = vec![ChatMessage::user_with_image(
        "beforeafter",
        make_test_png(8, 8),
        "before".len(),
    )];
    let mut polls = 0;
    let mut window_checked = false;
    let result = build_vision_request(
        &mut runtime,
        &config,
        &tiny_tokenizer(),
        &messages,
        0,
        true,
        false,
        &mut || {
            polls += 1;
            polls == 4
        },
        |_| {
            window_checked = true;
            Ok(())
        },
    )
    .expect("cancellation is not a worker failure");
    assert!(matches!(result, VisionRequestBuild::Cancelled));
    assert_eq!(polls, 4);
    assert!(window_checked);
}

/// The 21 `model.visual.*` tensor names `tiny_vision_config` (depth 1)
/// expects, matching `checkpoint.rs`'s `tensor_names` for the same
/// config shape.
fn tiny_vision_manifest_names() -> Vec<String> {
    let mut names = vec![
        "model.visual.patch_embed.proj.weight".to_string(),
        "model.visual.patch_embed.proj.bias".to_string(),
        "model.visual.pos_embed.weight".to_string(),
        "model.visual.merger.linear_fc1.weight".to_string(),
        "model.visual.merger.linear_fc1.bias".to_string(),
        "model.visual.merger.linear_fc2.weight".to_string(),
        "model.visual.merger.linear_fc2.bias".to_string(),
        "model.visual.merger.norm.weight".to_string(),
        "model.visual.merger.norm.bias".to_string(),
    ];
    for suffix in [
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    ] {
        names.push(format!("model.visual.blocks.0.{suffix}"));
    }
    names
}

#[test]
fn vision_request_build_rejects_oversized_image_with_dimensions_exceeded_code() {
    use axum::response::IntoResponse as _;

    let mut runtime = VisionRuntime::unsupported();
    let config = vision_build_config();
    // A thin (1px tall) image keeps the fixture tiny while still
    // tripping the serving-side hard pixel-dimension cap on width
    // alone -- deterministic regardless of any `LATTICE_VISION_MAX_PATCHES`
    // override in the test process's environment, unlike the
    // patch-count budget.
    let messages = vec![ChatMessage::user_with_image(
        "beforeafter",
        make_test_png(2049, 1),
        "before".len(),
    )];
    let err = build_vision_request(
        &mut runtime,
        &config,
        &tiny_tokenizer(),
        &messages,
        0,
        true,
        false,
        &mut || false,
        |_| Ok(()),
    )
    .expect_err("an oversized image must be rejected before any worker dispatch");
    match err {
        WorkerFailure::Rejected(api_err) => {
            assert_eq!(api_err.code(), "image_dimensions_exceeded");
            assert_eq!(
                api_err.into_response().status(),
                axum::http::StatusCode::BAD_REQUEST
            );
        }
        other => panic!("expected Rejected(image_dimensions_exceeded), got {other:?}"),
    }
}

#[test]
fn vision_request_build_rejects_malformed_image_bytes_with_invalid_image_code() {
    // Control: genuinely undecodable bytes must stay on the generic
    // `invalid_image` code, not the new `image_dimensions_exceeded`
    // code -- the split is scoped to well-formed-but-too-large images.
    use axum::response::IntoResponse as _;

    let mut runtime = VisionRuntime::unsupported();
    let config = vision_build_config();
    let messages = vec![ChatMessage::user_with_image(
        "beforeafter",
        b"not an image".to_vec(),
        "before".len(),
    )];
    let err = build_vision_request(
        &mut runtime,
        &config,
        &tiny_tokenizer(),
        &messages,
        0,
        true,
        false,
        &mut || false,
        |_| Ok(()),
    )
    .expect_err("malformed image bytes must be rejected");
    match err {
        WorkerFailure::Rejected(api_err) => {
            assert_eq!(api_err.code(), "invalid_image");
            assert_eq!(
                api_err.into_response().status(),
                axum::http::StatusCode::BAD_REQUEST
            );
        }
        other => panic!("expected Rejected(invalid_image), got {other:?}"),
    }
}

#[test]
fn vision_request_build_maps_a_lazy_load_failure_to_a_retryable_client_rejection() {
    use axum::response::IntoResponse as _;

    let temp = tempfile::tempdir().expect("tempdir");
    // Every manifest entry points at the same file, and that file is
    // not a valid tensor payload -- the capability preflight
    // (`validate_qwen35_vision_weight_inventory`) only checks that
    // every expected name occurs once and its file opens, so this
    // still advertises vision support; the actual lazy load fails
    // once a real image request triggers it.
    std::fs::write(temp.path().join("visual.bin"), b"not a valid tensor file")
        .expect("weight-source marker");
    let config = vision_build_config();
    let entries: Vec<_> = tiny_vision_manifest_names()
        .iter()
        .map(|name| serde_json::json!({"name": name, "file": "visual.bin"}))
        .collect();
    std::fs::write(
        temp.path().join("quantize_index.json"),
        serde_json::to_vec(&entries).expect("manifest fixture"),
    )
    .expect("test setup: write manifest");
    let mut runtime = VisionRuntime::from_model_config(temp.path().to_path_buf(), &config);
    assert!(
        runtime.is_supported(),
        "test setup: capability preflight must pass so the request reaches the lazy load"
    );

    let messages = vec![ChatMessage::user_with_image(
        "beforeafter",
        make_test_png(8, 8),
        "before".len(),
    )];
    let err = build_vision_request(
        &mut runtime,
        &config,
        &tiny_tokenizer(),
        &messages,
        0,
        true,
        false,
        &mut || false,
        |_| Ok(()),
    )
    .expect_err("a corrupt checkpoint must fail the lazy load");
    match err {
        WorkerFailure::Rejected(api_err) => {
            assert_eq!(api_err.code(), "vision_load_failed");
            let temp_path = temp.path().to_string_lossy().into_owned();
            assert!(
                !api_err.message().contains(&temp_path),
                "client-visible message must not leak the checkpoint path: {}",
                api_err.message()
            );
            assert_eq!(
                api_err.into_response().status(),
                axum::http::StatusCode::BAD_REQUEST
            );
        }
        other => panic!("expected Rejected(vision_load_failed), got {other:?}"),
    }
    assert!(
        runtime.is_supported(),
        "a lazy-load failure must stay retryable, not revoke capability permanently"
    );
}

#[test]
fn vision_runtime_capability_requires_config_token_metadata_and_weight_source() {
    let temp = tempfile::tempdir().expect("tempdir");
    std::fs::write(temp.path().join("quantize_index.json"), "[]").expect("weight-source marker");
    let mut config = Qwen35Config::qwen35_0_8b();
    assert!(!VisionRuntime::from_model_config(temp.path().to_path_buf(), &config).is_supported());
    config.vision_config = Some(tiny_vision_config(config.hidden_size));
    config.image_token_id = Some(10);
    config.vision_start_token_id = Some(11);
    config.vision_end_token_id = Some(12);
    assert!(
        !VisionRuntime::from_model_config(temp.path().to_path_buf(), &config).is_supported(),
        "an empty manifest must not advertise vision capability"
    );

    let mut names = vec![
        "model.visual.patch_embed.proj.weight".to_string(),
        "model.visual.patch_embed.proj.bias".to_string(),
        "model.visual.pos_embed.weight".to_string(),
        "model.visual.merger.linear_fc1.weight".to_string(),
        "model.visual.merger.linear_fc1.bias".to_string(),
        "model.visual.merger.linear_fc2.weight".to_string(),
        "model.visual.merger.linear_fc2.bias".to_string(),
        "model.visual.merger.norm.weight".to_string(),
        "model.visual.merger.norm.bias".to_string(),
    ];
    for suffix in [
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    ] {
        names.push(format!("model.visual.blocks.0.{suffix}"));
    }
    std::fs::write(temp.path().join("visual.bin"), b"inventory preflight")
        .expect("visual tensor marker");
    let entries: Vec<_> = names
        .iter()
        .map(|name| serde_json::json!({"name": name, "file": "visual.bin"}))
        .collect();
    std::fs::write(
        temp.path().join("quantize_index.json"),
        serde_json::to_vec(&entries).expect("manifest fixture"),
    )
    .expect("complete vision manifest");
    let mut invalid_tokens = config.clone();
    invalid_tokens.image_token_id = Some(config.vocab_size as u32);
    assert!(
        !VisionRuntime::from_model_config(temp.path().to_path_buf(), &invalid_tokens)
            .is_supported(),
        "out-of-vocabulary image metadata must not advertise capability"
    );
    invalid_tokens.image_token_id = invalid_tokens.vision_start_token_id;
    assert!(
        !VisionRuntime::from_model_config(temp.path().to_path_buf(), &invalid_tokens)
            .is_supported(),
        "aliased vision token metadata must not advertise capability"
    );
    let mut runtime = VisionRuntime::from_model_config(temp.path().to_path_buf(), &config);
    assert!(runtime.is_supported());
    let (job_tx, _job_rx) = mpsc::unbounded_channel();
    let client = MetalWorkerClient::with_owner(
        job_tx,
        Arc::new(Semaphore::new(1)),
        runtime.shared_capability(),
        MetalWorkerOwner::unattached_for_test(),
    );
    assert!(client.supports_vision());
    let cancelled = runtime
        .get_or_load(&mut || true)
        .expect("cancellation is not a lazy-load failure");
    assert!(matches!(cancelled, VisionRuntimeLoad::Cancelled));
    assert!(runtime.is_supported());
    assert!(
        client.supports_vision(),
        "cancellation must preserve Pending capability for a later retry"
    );
    let mut never_cancel = || false;
    let first_error = runtime
        .get_or_load(&mut never_cancel)
        .expect_err("junk tensor payload must fail its first lazy load");
    assert!(first_error.contains("vision weights failed to load"));
    assert!(
        runtime.is_supported(),
        "a lazy-load failure must stay retryable, not revoke capability permanently"
    );
    assert!(
        client.supports_vision(),
        "a lazy-load failure must not revoke the shared client capability -- the \
             caller can retry"
    );

    // Fix the checkpoint in place: write a genuinely valid q4 tensor for
    // every `model.visual.*` name `config`'s tiny vision config expects,
    // then retry. Reaching `VisionRuntimeLoad::Ready` (not another
    // error) is the proof the second call actually attempted the load
    // again rather than replaying a cached failure.
    let hidden = 8usize;
    let qkv_out = 3 * hidden;
    let mlp_intermediate = 4 * hidden;
    let merge_in = 2 * 2 * hidden;
    let out_hidden = config.hidden_size;
    let mut valid_shapes: Vec<(String, Vec<usize>)> = vec![
        (
            "model.visual.patch_embed.proj.weight".to_string(),
            vec![hidden, 3, 1, 2, 2],
        ),
        (
            "model.visual.patch_embed.proj.bias".to_string(),
            vec![hidden],
        ),
        (
            "model.visual.pos_embed.weight".to_string(),
            vec![16, hidden],
        ),
        (
            "model.visual.merger.linear_fc1.weight".to_string(),
            vec![merge_in, merge_in],
        ),
        (
            "model.visual.merger.linear_fc1.bias".to_string(),
            vec![merge_in],
        ),
        (
            "model.visual.merger.linear_fc2.weight".to_string(),
            vec![out_hidden, merge_in],
        ),
        (
            "model.visual.merger.linear_fc2.bias".to_string(),
            vec![out_hidden],
        ),
        ("model.visual.merger.norm.weight".to_string(), vec![hidden]),
        ("model.visual.merger.norm.bias".to_string(), vec![hidden]),
    ];
    for (suffix, shape) in [
        ("attn.qkv.weight", vec![qkv_out, hidden]),
        ("attn.qkv.bias", vec![qkv_out]),
        ("attn.proj.weight", vec![hidden, hidden]),
        ("attn.proj.bias", vec![hidden]),
        ("mlp.linear_fc1.weight", vec![mlp_intermediate, hidden]),
        ("mlp.linear_fc1.bias", vec![mlp_intermediate]),
        ("mlp.linear_fc2.weight", vec![hidden, mlp_intermediate]),
        ("mlp.linear_fc2.bias", vec![hidden]),
        ("norm1.weight", vec![hidden]),
        ("norm1.bias", vec![hidden]),
        ("norm2.weight", vec![hidden]),
        ("norm2.bias", vec![hidden]),
    ] {
        valid_shapes.push((format!("model.visual.blocks.0.{suffix}"), shape));
    }
    let mut manifest_entries = Vec::new();
    for (i, (name, shape)) in valid_shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&vec![0.25_f64; numel], shape)
            .expect("quantize succeeds");
        let file_name = format!("retry{i}.q4");
        crate::weights::q4_weights::save_q4_file(&temp.path().join(&file_name), &q4)
            .expect("test setup: write valid q4 file");
        manifest_entries.push(format!(
            r#"{{"name":"{name}","file":"{file_name}","quantized":true}}"#
        ));
    }
    std::fs::write(
        temp.path().join("quantize_index.json"),
        format!("[{}]", manifest_entries.join(",")),
    )
    .expect("test setup: overwrite manifest with a valid checkpoint");

    let retried = runtime.get_or_load(&mut never_cancel).expect(
        "a fixed checkpoint must load successfully on retry, not replay the cached failure",
    );
    assert!(matches!(retried, VisionRuntimeLoad::Ready(_)));
    assert!(runtime.is_supported());
    assert!(client.supports_vision());
    assert!(
        !VisionRuntime::from_model_config(PathBuf::from("/unused"), &config).is_supported(),
        "config metadata alone must not advertise vision without a supported weight source"
    );
    config.vision_end_token_id = None;
    assert!(!VisionRuntime::from_model_config(temp.path().to_path_buf(), &config).is_supported());
}

/// A `Pending` runtime whose weight-source manifest passes the inventory
/// preflight (every expected tensor name is present) but whose backing
/// file is not real tensor data, so any actual load attempt fails —
/// mirrors the junk-payload fixture in
/// `vision_runtime_capability_requires_config_token_metadata_and_weight_source`.
/// Reused by the `--preload-vision` tests below, which care about the
/// *failure* path, not a successful load (a real checkpoint is needed
/// for that and is already covered by the `metal_qwen35` gate tests).
fn pending_vision_runtime_with_unloadable_weights() -> (tempfile::TempDir, VisionRuntime) {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut config = Qwen35Config::qwen35_0_8b();
    config.vision_config = Some(tiny_vision_config(config.hidden_size));
    config.image_token_id = Some(10);
    config.vision_start_token_id = Some(11);
    config.vision_end_token_id = Some(12);

    let mut names = vec![
        "model.visual.patch_embed.proj.weight".to_string(),
        "model.visual.patch_embed.proj.bias".to_string(),
        "model.visual.pos_embed.weight".to_string(),
        "model.visual.merger.linear_fc1.weight".to_string(),
        "model.visual.merger.linear_fc1.bias".to_string(),
        "model.visual.merger.linear_fc2.weight".to_string(),
        "model.visual.merger.linear_fc2.bias".to_string(),
        "model.visual.merger.norm.weight".to_string(),
        "model.visual.merger.norm.bias".to_string(),
    ];
    for suffix in [
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    ] {
        names.push(format!("model.visual.blocks.0.{suffix}"));
    }
    std::fs::write(temp.path().join("visual.bin"), b"not real tensor data")
        .expect("visual tensor marker");
    let entries: Vec<_> = names
        .iter()
        .map(|name| serde_json::json!({"name": name, "file": "visual.bin"}))
        .collect();
    std::fs::write(
        temp.path().join("quantize_index.json"),
        serde_json::to_vec(&entries).expect("manifest fixture"),
    )
    .expect("complete vision manifest");

    let runtime = VisionRuntime::from_model_config(temp.path().to_path_buf(), &config);
    assert!(
        runtime.is_supported(),
        "fixture must advertise capability so preload has a Pending state to act on"
    );
    (temp, runtime)
}

/// `--preload-vision` on a checkpoint with no vision capability (or an
/// already-resolved runtime) must not error and must not change
/// capability — there is nothing pending to load.
#[test]
fn vision_runtime_preload_is_noop_when_unsupported() {
    let mut runtime = VisionRuntime::unsupported();
    runtime
        .preload()
        .expect("preloading an unsupported runtime is a no-op, not an error");
    assert!(!runtime.is_supported());
}

/// The failure-fallback contract `--preload-vision` depends on: a failed
/// eager load at startup must warn (the CLI layer's job) and leave the
/// runtime exactly as if `--preload-vision` had never been passed — still
/// `Pending`, still advertising capability, so the first real image
/// request gets its own normal lazy-load attempt rather than an
/// immediately-terminal one.
///
/// Mutation-sensitivity: this is the test that reddens if `preload()`
/// propagates its error in a way that revokes vision capability instead
/// of leaving `Pending` on failure. Both `preload` and `get_or_load` now
/// leave a failed load retryable, so the second half of this test pins
/// that shared policy: one unloadable fixture must not disable vision for
/// the rest of the session through either entry point.
#[test]
fn vision_runtime_preload_failure_falls_back_to_lazy_pending_state() {
    let (_temp, mut runtime) = pending_vision_runtime_with_unloadable_weights();

    let preload_err = runtime
        .preload()
        .expect_err("junk tensor payload must fail preload");
    assert!(preload_err.contains("vision preload failed"));
    assert!(
        runtime.is_supported(),
        "a failed preload must not revoke vision capability — the CLI falls back to lazy \
             loading, it does not disable vision for the session"
    );

    // The first real image request's lazy load must still be attempted
    // fresh, not short-circuited by state left behind by preload — same
    // junk payload, so it fails too, and it fails through `get_or_load`'s
    // own error path rather than preload's.
    let mut never_cancel = || false;
    let lazy_err = runtime
        .get_or_load(&mut never_cancel)
        .expect_err("lazy load retries the same unloadable fixture and fails the same way");
    assert!(lazy_err.contains("vision weights failed to load"));
    assert!(
        runtime.is_supported(),
        "a failed lazy load leaves state at `Pending` and `vision_supported` untouched, so a \
             later request retries rather than being rejected forever from one transient failure"
    );
}

// ── GPU-free fakes, ported from lattice_serve.rs's pre-existing
//    `run_worker_loop` test suite (#832 migrates them here) ──────────

#[allow(clippy::type_complexity)]
fn fake_generate(
    cap: usize,
    started: Arc<AtomicUsize>,
    ran_tokens: Arc<AtomicUsize>,
) -> impl FnMut(
    &[ChatMessage],
    &GenerateConfig,
    bool,
    &mut dyn FnMut(&str, u32) -> bool,
    &mut dyn FnMut() -> bool,
) -> Result<GenerateOutput, WorkerFailure> {
    move |_messages, _cfg, _explicitly_thinking, on_token, should_cancel| {
        started.fetch_add(1, Ordering::SeqCst);
        let mut n = 0usize;
        for i in 0..cap {
            std::thread::sleep(Duration::from_millis(5));
            if should_cancel() {
                break;
            }
            if !on_token("x", i as u32) {
                break;
            }
            n += 1;
            ran_tokens.fetch_add(1, Ordering::SeqCst);
        }
        Ok(GenerateOutput {
            text: "x".repeat(n),
            token_ids: vec![0; n],
            prompt_tokens: 1,
            generated_tokens: n,
            stopped: false,
            stop_reason: None,
            token_logprobs: vec![],
        })
    }
}

#[allow(clippy::type_complexity)]
fn fake_generate_with_prefill_gap(
    prefill_steps: usize,
    decode_cap: usize,
    entered_decode: Arc<AtomicBool>,
) -> impl FnMut(
    &[ChatMessage],
    &GenerateConfig,
    bool,
    &mut dyn FnMut(&str, u32) -> bool,
    &mut dyn FnMut() -> bool,
) -> Result<GenerateOutput, WorkerFailure> {
    move |_messages, _cfg, _explicitly_thinking, on_token, should_cancel| {
        for _ in 0..prefill_steps {
            std::thread::sleep(Duration::from_millis(5));
            if should_cancel() {
                return Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: 1,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: None,
                    token_logprobs: vec![],
                });
            }
        }
        entered_decode.store(true, Ordering::SeqCst);
        let mut n = 0usize;
        for i in 0..decode_cap {
            std::thread::sleep(Duration::from_millis(5));
            if should_cancel() {
                break;
            }
            if !on_token("x", i as u32) {
                break;
            }
            n += 1;
        }
        Ok(GenerateOutput {
            text: "x".repeat(n),
            token_ids: vec![0; n],
            prompt_tokens: 1,
            generated_tokens: n,
            stopped: false,
            stop_reason: None,
            token_logprobs: vec![],
        })
    }
}

#[allow(clippy::type_complexity)]
fn fake_generate_fails_once_then_succeeds(
    message: &'static str,
    call_count: Arc<AtomicUsize>,
) -> impl FnMut(
    &[ChatMessage],
    &GenerateConfig,
    bool,
    &mut dyn FnMut(&str, u32) -> bool,
    &mut dyn FnMut() -> bool,
) -> Result<GenerateOutput, WorkerFailure> {
    move |_messages, _cfg, _explicitly_thinking, on_token, _should_cancel| {
        if call_count.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(WorkerFailure::Failed(message.to_string()));
        }
        let _ = on_token("x", 0);
        Ok(GenerateOutput {
            text: "x".to_string(),
            token_ids: vec![0],
            prompt_tokens: 1,
            generated_tokens: 1,
            stopped: true,
            stop_reason: None,
            token_logprobs: vec![],
        })
    }
}

/// Builds a `WorkerJob` plus the receiver its worker replies on and the
/// guard that cancels it when dropped (the same guard a real handler
/// moves into the SSE stream / keeps local for non-streaming, standing
/// in here for "the client is still connected").
/// Returns the job already wrapped in its [`WorkerMessage`], because
/// every caller sends it straight down the queue and none of them looks
/// inside.
fn make_job() -> (
    WorkerMessage,
    mpsc::UnboundedReceiver<WorkerEvent>,
    crate::serve::CancelOnDrop,
) {
    let (tx, rx) = mpsc::unbounded_channel::<WorkerEvent>();
    let (cancel_guard, cancel_rx) = crate::serve::cancel_pair();
    // These FIFO/cancellation-loop tests drive `WorkerJob` directly
    // (bypassing `MetalWorkerClient::submit`'s admission check
    // entirely), so each job gets its own throwaway one-permit
    // semaphore rather than sharing a real admission cap -- these tests
    // are not exercising #932's admission behavior at all.
    let permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("fresh single-permit semaphore must have a permit available");
    let job = WorkerJob {
        messages: vec![ChatMessage::user("hi")],
        cfg: GenerateConfig::default(),
        lora: Vec::new(),
        explicitly_thinking: false,
        stream: false,
        tx,
        cancel: cancel_rx,
        _admission_permit: permit,
    };
    (WorkerMessage::Generate(job), rx, cancel_guard)
}

// ─── adapter control messages ────────────────────────────────────────

/// A generator that reports the value of a shared counter as each token,
/// pauses after its first token until released, and then finishes.
///
/// The counter stands in for "which adapter is loaded". If an adapter
/// command were ever applied while this generator is on the stack, the
/// tokens emitted after the pause would carry a different value from the
/// ones emitted before it, and the assembled output would say so. A
/// generator that could not express that difference would make the
/// ordering test unfalsifiable.
#[allow(clippy::type_complexity)]
fn epoch_reporting_generate(
    epoch: Arc<AtomicUsize>,
    tokens: usize,
    started: std::sync::mpsc::SyncSender<()>,
    release: Arc<Mutex<Option<std::sync::mpsc::Receiver<()>>>>,
) -> impl FnMut(
    &[ChatMessage],
    &GenerateConfig,
    bool,
    &mut dyn FnMut(&str, u32) -> bool,
    &mut dyn FnMut() -> bool,
) -> Result<GenerateOutput, WorkerFailure> {
    move |_messages, _cfg, _explicitly_thinking, on_token, should_cancel| {
        let mut text = String::new();
        for i in 0..tokens {
            // Polled every iteration, as this module's contract for an
            // injected generator requires. It is also the call site any
            // mid-generation channel pump would have to live on, so a
            // generator that skipped it would make the ordering mutation
            // unobservable -- which is exactly what happened on the first
            // attempt at this test.
            if should_cancel() {
                break;
            }
            let delta = epoch.load(Ordering::SeqCst).to_string();
            text.push_str(&delta);
            on_token(&delta, i as u32);
            if i == 0 {
                let _ = started.try_send(());
                if let Some(rx) = lock_unpoisoned(&release).take() {
                    // Block here long enough that a control message sent
                    // by the test is definitely sitting in the queue
                    // while this generation is still running.
                    let _ = rx.recv();
                }
            }
        }
        let generated_tokens = text.len();
        Ok(GenerateOutput {
            text,
            token_ids: vec![0; generated_tokens],
            prompt_tokens: 1,
            generated_tokens,
            stopped: false,
            stop_reason: None,
            token_logprobs: vec![],
        })
    }
}

/// Drive one generation to completion through the real loop, optionally
/// injecting an adapter command while that generation is mid-flight.
/// Returns the assembled output text and the epoch value afterwards.
fn run_one_generation(inject_control: bool) -> (String, usize) {
    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let epoch = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel::<()>(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel::<()>(1);
    let release = Arc::new(Mutex::new(Some(release_rx)));

    let (job, mut events, _guard) = make_job();
    msg_tx.send(job).unwrap();

    let epoch_for_generate = epoch.clone();
    let epoch_for_control = epoch.clone();
    let handle = std::thread::spawn(move || {
        run_worker_loop_with_control(
            msg_rx,
            epoch_reporting_generate(epoch_for_generate, 6, started_tx, release),
            move |command| {
                assert!(matches!(command, AdapterCommand::Unload { id: 0 }));
                epoch_for_control.fetch_add(1, Ordering::SeqCst);
                Ok(AdapterControlResult::Unloaded(0))
            },
        );
    });

    started_rx.recv().expect("generation should start");
    let control_reply = if inject_control {
        let (reply, reply_rx) = oneshot::channel();
        let permit = Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .expect("fresh semaphore");
        msg_tx
            .send(WorkerMessage::Control(ControlRequest {
                command: AdapterCommand::Unload { id: 0 },
                reply,
                _control_permit: permit,
            }))
            .unwrap();
        Some(reply_rx)
    } else {
        None
    };
    let _ = release_tx.send(());

    let mut text = String::new();
    loop {
        match events.blocking_recv() {
            Some(WorkerEvent::Delta(d)) => text.push_str(&d),
            Some(WorkerEvent::Complete(_)) | None => break,
            Some(other) => panic!("unexpected event: {other:?}"),
        }
    }

    if let Some(reply_rx) = control_reply {
        // The command must actually have been applied, not silently
        // dropped -- otherwise "the generation was unaffected" would be
        // true for the uninteresting reason.
        let outcome = reply_rx.blocking_recv().expect("control reply channel");
        assert_eq!(
            outcome,
            Ok(AdapterControlResult::Unloaded(0)),
            "the injected command must have run"
        );
    }

    drop(msg_tx);
    handle.join().unwrap();
    let final_epoch = epoch.load(Ordering::SeqCst);
    (text, final_epoch)
}

/// An adapter command submitted while a generation is running does not
/// change that generation's output.
///
/// This asserts the generation is byte-identical to the same generation
/// run with no command in flight. A weaker test -- send both, assert no
/// error -- passes against a worker that applies the command mid-decode,
/// which is the exact failure this design exists to prevent.
#[test]
fn adapter_command_during_generation_does_not_change_that_generation() {
    let (baseline_text, baseline_epoch) = run_one_generation(false);
    let (concurrent_text, concurrent_epoch) = run_one_generation(true);

    assert_eq!(
        baseline_epoch, 0,
        "the baseline run must not apply any command"
    );
    assert_eq!(
        concurrent_epoch, 1,
        "the injected command must have been applied after the generation"
    );
    assert_eq!(
        concurrent_text, baseline_text,
        "a command queued mid-generation must not reach that generation: \
             baseline {baseline_text:?} vs concurrent {concurrent_text:?}"
    );
    assert_eq!(
        baseline_text, "000000",
        "sanity: the baseline reports epoch 0 for every token"
    );
}

/// The handler's error string reaches the caller unchanged.
///
/// Across a channel that string is the entire diagnosis the caller gets,
/// so a loop that collapsed it to a bool, or wrapped it in wording of its
/// own, would leave the caller unable to tell "already loaded" from
/// "rank mismatch".
#[test]
fn control_error_string_crosses_the_channel_verbatim() {
    const LOADER_MESSAGE: &str = "LoRA adapter already loaded; call unload_lora_adapter first";
    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let handle = std::thread::spawn(move || {
        run_worker_loop_with_control(
            msg_rx,
            fake_generate(
                1,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            ),
            |_command| {
                Err(AdapterControlError::InvalidAdapter(
                    LOADER_MESSAGE.to_string(),
                ))
            },
        );
    });

    let (reply, reply_rx) = oneshot::channel();
    let permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("fresh semaphore");
    msg_tx
        .send(WorkerMessage::Control(ControlRequest {
            command: AdapterCommand::Unload { id: 0 },
            reply,
            _control_permit: permit,
        }))
        .unwrap();

    let outcome = reply_rx.blocking_recv().expect("control reply channel");
    assert_eq!(
        outcome,
        Err(AdapterControlError::InvalidAdapter(
            LOADER_MESSAGE.to_string()
        ))
    );

    drop(msg_tx);
    handle.join().unwrap();
}

/// A second adapter command, while the first is still in flight, is
/// refused rather than queued.
///
/// The client here has no worker draining its queue, so the first
/// request stays in flight and keeps its permit, which is precisely the
/// state the refusal is about.
#[test]
fn second_in_flight_adapter_command_is_refused() {
    let (client, _msg_rx) = test_client_and_jobs();

    let first = client
        .submit_adapter_command(AdapterCommand::Unload { id: 0 })
        .expect("the first command should be admitted");

    let second = client.submit_adapter_command(AdapterCommand::Unload { id: 0 });
    match second {
        Err(ApiError::ServiceUnavailable { ref message }) => {
            assert!(
                message.contains("already in flight"),
                "the refusal should say why, got: {message}"
            );
        }
        other => panic!("a second in-flight command must be refused, got {other:?}"),
    }

    // Generation admission is a different semaphore and is untouched by
    // the control refusal: a stuck adapter command must not stop the
    // model serving.
    client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            crate::serve::cancel_pair().1,
        )
        .expect("generation admission is independent of control admission");

    // The permit is not released by dropping the caller's reply
    // receiver: it lives in the `ControlRequest` still sitting in the
    // queue, and is released when the worker finishes with it. Nothing
    // here drains that queue, so there is no "and now it is admitted
    // again" arm to assert -- the arm that exists is the one above.
    drop(first);
}

#[test]
fn queued_job_cancelled_before_dequeue_sends_exactly_one_cancelled_event() {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));

    // Job 1 occupies the worker (50 fake tokens, 5ms apart = ~250ms)
    // long enough that job 2 is still sitting in the queue, untouched,
    // when we cancel it a few lines down.
    let (job1, rx1, _guard1) = make_job();
    job_tx.send(job1).unwrap();

    // Job 2: cancelled client-side (guard dropped) immediately, while
    // it is still queued behind job 1.
    let (job2, mut rx2, guard2) = make_job();
    job_tx.send(job2).unwrap();
    drop(guard2);

    // Job 3: submitted after the cancelled one, to prove the worker
    // moves on and keeps serving correctly afterward.
    let (job3, rx3, _guard3) = make_job();
    job_tx.send(job3).unwrap();
    drop(job_tx);

    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let handle =
        std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(50, started2, ran2)));

    let completion_tokens_of = |mut rx: mpsc::UnboundedReceiver<WorkerEvent>| -> Option<usize> {
        let mut ct = None;
        while let Some(ev) = rx.blocking_recv() {
            if let WorkerEvent::Complete(output) = ev {
                ct = Some(output.generated_tokens);
            }
        }
        ct
    };

    assert_eq!(
        completion_tokens_of(rx1),
        Some(50),
        "job 1 should run to completion undisturbed"
    );

    // Job 2 must produce exactly one event: Cancelled -- the single
    // shared contract (#832) this refactor picks, replacing both
    // binaries' prior divergent behavior (an empty interrupted
    // GenerateOutput reply vs. total silence).
    match rx2.blocking_recv() {
        Some(WorkerEvent::Cancelled) => {}
        other => panic!("expected exactly one Cancelled event, got {other:?}"),
    }
    assert!(
        rx2.blocking_recv().is_none(),
        "cancelled queued job must produce no further events after Cancelled"
    );

    assert_eq!(
        completion_tokens_of(rx3),
        Some(50),
        "worker must survive cancelling job 2 and serve job 3 normally afterward"
    );

    handle.join().expect("worker thread must not panic");

    assert_eq!(
        started.load(Ordering::SeqCst),
        2,
        "generate() must run exactly twice (job 1, job 3) -- never for cancelled job 2"
    );
    assert_eq!(
        ran_tokens.load(Ordering::SeqCst),
        100,
        "50 real fake-tokens each for job 1 and job 3, zero for cancelled job 2"
    );
}

#[test]
fn job_whose_event_receiver_is_already_closed_is_cancelled_without_running_generate() {
    // Distinct from a `cancel`-guard drop: this job's `cancel` watch
    // stays `false` forever (the guard is kept alive), but its event
    // receiver is dropped before the worker ever dequeues it. The
    // dequeue-time check must catch this independently (#832: "cancel
    // OR event_receiver_closed").
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let (tx, rx) = mpsc::unbounded_channel::<WorkerEvent>();
    drop(rx);
    let (_guard, cancel_rx) = crate::serve::cancel_pair();
    let permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("fresh single-permit semaphore must have a permit available");
    let job = WorkerJob {
        messages: vec![ChatMessage::user("hi")],
        cfg: GenerateConfig::default(),
        lora: Vec::new(),
        explicitly_thinking: false,
        stream: false,
        tx,
        cancel: cancel_rx,
        _admission_permit: permit,
    };
    job_tx.send(WorkerMessage::Generate(job)).unwrap();
    drop(job_tx);

    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));
    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let handle =
        std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(50, started2, ran2)));
    handle.join().expect("worker thread must not panic");

    assert_eq!(
        started.load(Ordering::SeqCst),
        0,
        "generate() must never run for a job whose event receiver was already closed"
    );
    assert_eq!(ran_tokens.load(Ordering::SeqCst), 0);
}

#[test]
fn running_job_cancelled_midstream_stops_early_and_worker_survives() {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));

    let (job1, mut rx1, guard1) = make_job();
    job_tx.send(job1).unwrap();
    let mut guard1 = Some(guard1);

    let (job2, mut rx2, _guard2) = make_job();
    job_tx.send(job2).unwrap();
    drop(job_tx);

    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let handle =
        std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(2000, started2, ran2)));

    let mut seen = 0;
    loop {
        match rx1.blocking_recv() {
            Some(WorkerEvent::Delta(_)) => {
                seen += 1;
                if seen == 5 {
                    guard1.take();
                }
            }
            Some(WorkerEvent::Complete(output)) => {
                assert!(
                    output.generated_tokens < 2000,
                    "job 1 must stop well short of its 2000-token cap after \
                         cancellation, got {}",
                    output.generated_tokens
                );
                assert!(
                    output.generated_tokens < 100,
                    "job 1 must stop within a handful of tokens of the client \
                         disconnecting, not run on regardless; got {}",
                    output.generated_tokens
                );
                break;
            }
            Some(WorkerEvent::Failed(message)) => {
                panic!("fake_generate never fails; unexpected Failed: {message}")
            }
            Some(WorkerEvent::ConstraintBlocked(message)) => {
                panic!(
                    "fake_generate never blocks on a grammar constraint; unexpected \
                         ConstraintBlocked: {message}"
                )
            }
            Some(WorkerEvent::Rejected(err)) => {
                panic!("fake_generate never rejects; unexpected Rejected: {err:?}")
            }
            Some(WorkerEvent::Cancelled) => {
                panic!("job 1 was already running -- Cancelled is a dequeue-only event")
            }
            None => panic!("job 1's reply channel closed before a Complete event"),
        }
    }

    let mut n2 = None;
    while let Some(ev) = rx2.blocking_recv() {
        if let WorkerEvent::Complete(output) = ev {
            n2 = Some(output.generated_tokens);
        }
    }
    assert_eq!(
        n2,
        Some(2000),
        "worker must survive mid-stream cancellation and serve the next job to completion"
    );

    handle.join().expect("worker thread must not panic");
}

#[test]
fn running_job_cancelled_during_prefill_like_phase_never_calls_on_token() {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let entered_decode = Arc::new(AtomicBool::new(false));

    let (job1, mut rx1, guard1) = make_job();
    job_tx.send(job1).unwrap();
    job_tx.send(make_job().0).unwrap_or(()); // keep queue non-trivial; unused receiver dropped
    drop(job_tx);

    let entered2 = entered_decode.clone();
    let handle = std::thread::spawn(move || {
        run_worker_loop(job_rx, fake_generate_with_prefill_gap(400, 50, entered2))
    });

    std::thread::sleep(Duration::from_millis(20));
    drop(guard1);

    match rx1.blocking_recv() {
        Some(WorkerEvent::Delta(_)) => panic!(
            "on_token must never be called: cancellation happened while the fake \
                 generator was still in its prefill-like phase, which does not call \
                 on_token at all"
        ),
        Some(WorkerEvent::Complete(output)) => {
            assert_eq!(
                output.generated_tokens, 0,
                "job cancelled during the prefill-like phase must produce zero tokens, \
                     got {}",
                output.generated_tokens
            );
        }
        Some(WorkerEvent::Failed(message)) => {
            panic!("fake_generate_with_prefill_gap never fails; unexpected Failed: {message}")
        }
        Some(WorkerEvent::ConstraintBlocked(message)) => {
            panic!(
                "fake_generate_with_prefill_gap never blocks on a grammar constraint; \
                     unexpected ConstraintBlocked: {message}"
            )
        }
        Some(WorkerEvent::Rejected(err)) => {
            panic!("fake_generate_with_prefill_gap never rejects; unexpected Rejected: {err:?}")
        }
        Some(WorkerEvent::Cancelled) => {
            panic!("job 1 was already dequeued and running -- not a dequeue-time cancel")
        }
        None => panic!("job 1's reply channel closed before a Complete event"),
    }

    handle.join().expect("worker thread must not panic");

    assert!(
        !entered_decode.load(Ordering::SeqCst),
        "should_cancel alone (on_token is never called during this phase) must stop \
             the job before the decode phase is ever reached"
    );
}

#[test]
fn running_job_whose_event_receiver_closes_during_prefill_like_phase_never_calls_on_token() {
    // Distinct from the dequeue-time receiver test (the receiver is open
    // when job 1 is dequeued) and from the flag-driven prefill test (job
    // 1's cancel guard is kept alive for the whole test, so the cancel
    // flag stays unset through the observed generation; the guards drop
    // at scope exit, after the worker is joined). The only thing that
    // can stop job 1 inside its prefill-like phase is `should_cancel`
    // observing the closed receiver.
    //
    // The receiver is dropped only after `generate` has been entered, and
    // the generator holds its first predicate poll until that drop has
    // happened, so the first poll is causally after the close and no
    // sleep orders anything. Every wait below is bounded except the final
    // join, which cannot hang the test: it follows the done signal, and
    // that is sent only after the worker loop has returned.
    const WAIT: Duration = Duration::from_secs(10);

    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel::<()>(1);
    let (closed_tx, closed_rx) = std::sync::mpsc::sync_channel::<()>(1);
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel::<()>(1);
    let entered_decode = Arc::new(AtomicBool::new(false));
    let on_token_calls = Arc::new(AtomicUsize::new(0));
    let job1_polls = Arc::new(Mutex::new(Vec::<bool>::new()));
    let ran_tokens = Arc::new(AtomicUsize::new(0));

    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let (job1, rx1, _guard1) = make_job();
    let (job2, mut rx2, _guard2) = make_job();
    job_tx.send(job1).unwrap();
    job_tx.send(job2).unwrap();
    drop(job_tx);

    let handle = std::thread::spawn({
        let entered_decode = entered_decode.clone();
        let on_token_calls = on_token_calls.clone();
        let job1_polls = job1_polls.clone();
        let ran_tokens = ran_tokens.clone();
        move || {
            let mut prefill_gap = fake_generate_with_prefill_gap(50, 50, entered_decode);
            let mut follow_up = fake_generate(50, Arc::new(AtomicUsize::new(0)), ran_tokens);
            let mut calls = 0usize;
            run_worker_loop(
                job_rx,
                move |messages, cfg, explicitly_thinking, on_token, should_cancel| {
                    calls += 1;
                    if calls > 1 {
                        return follow_up(
                            messages,
                            cfg,
                            explicitly_thinking,
                            on_token,
                            should_cancel,
                        );
                    }
                    let _ = entered_tx.try_send(());
                    if closed_rx.recv_timeout(WAIT).is_err() {
                        return Err(WorkerFailure::Failed(
                            "the test never closed job 1's event receiver".to_string(),
                        ));
                    }
                    let mut counted_on_token = |delta: &str, token_id: u32| {
                        on_token_calls.fetch_add(1, Ordering::SeqCst);
                        on_token(delta, token_id)
                    };
                    let mut recorded_should_cancel = || {
                        let cancelled = should_cancel();
                        lock_unpoisoned(&job1_polls).push(cancelled);
                        cancelled
                    };
                    prefill_gap(
                        messages,
                        cfg,
                        explicitly_thinking,
                        &mut counted_on_token,
                        &mut recorded_should_cancel,
                    )
                },
            );
            let _ = done_tx.send(());
        }
    });

    entered_rx
        .recv_timeout(WAIT)
        .expect("generate must be entered for job 1 before its event receiver is closed");
    drop(rx1);
    closed_tx
        .send(())
        .expect("the generator must still be waiting for job 1's receiver to close");
    done_rx
        .recv_timeout(WAIT)
        .expect("the worker must finish both jobs once job 1's receiver is closed");
    handle.join().expect("worker thread must not panic");

    assert_eq!(
        *lock_unpoisoned(&job1_polls),
        vec![true],
        "the first predicate poll after the event receiver closed must report \
             cancellation, and the generator must stop at that poll"
    );
    assert_eq!(
        on_token_calls.load(Ordering::SeqCst),
        0,
        "on_token must never be called for a job whose receiver closed during the \
             prefill-like phase"
    );
    assert!(
        !entered_decode.load(Ordering::SeqCst),
        "a closed event receiver alone must stop job 1 before the decode phase is reached"
    );

    let mut job2_tokens = None;
    while let Some(event) = rx2.blocking_recv() {
        if let WorkerEvent::Complete(output) = event {
            job2_tokens = Some(output.generated_tokens);
        }
    }
    assert_eq!(
        job2_tokens,
        Some(50),
        "worker must survive job 1 stopping on a closed receiver and serve job 2 to \
             completion"
    );
    assert_eq!(
        ran_tokens.load(Ordering::SeqCst),
        50,
        "job 2 must run its 50 tokens; job 1 contributes none"
    );
}

#[test]
fn generation_failure_is_reported_as_failed_not_complete() {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();

    let (job1, mut rx1, _guard1) = make_job();
    job_tx.send(job1).unwrap();
    let (job2, mut rx2, _guard2) = make_job();
    job_tx.send(job2).unwrap();
    drop(job_tx);

    let call_count = Arc::new(AtomicUsize::new(0));
    let handle = std::thread::spawn({
        let call_count = call_count.clone();
        move || {
            run_worker_loop(
                job_rx,
                fake_generate_fails_once_then_succeeds(
                    "grammar constraint blocked every token; no legal continuation \
                         exists in the current grammar state",
                    call_count,
                ),
            )
        }
    });

    match rx1.blocking_recv() {
        Some(WorkerEvent::Failed(message)) => {
            assert!(
                message.contains("grammar constraint blocked every token"),
                "Failed must carry the underlying error message, got: {message}"
            );
        }
        Some(WorkerEvent::Complete(_)) => panic!(
            "a failed generation must never be reported as Complete -- that would \
                 silently hand the HTTP layer a fabricated result for a request that \
                 produced no legal output"
        ),
        other => panic!("expected Failed as the first and only event, got {other:?}"),
    }

    let mut done = None;
    while let Some(ev) = rx2.blocking_recv() {
        if let WorkerEvent::Complete(output) = ev {
            done = Some(output.generated_tokens);
        }
    }
    assert_eq!(
        done,
        Some(1),
        "worker thread must survive a failed generation and serve the next job \
             normally afterward"
    );

    handle
        .join()
        .expect("worker thread must not panic on a generation error");
}

#[test]
fn queue_closure_lets_the_worker_thread_exit_and_join() {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));
    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let handle =
        std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(1, started2, ran2)));
    // No jobs submitted at all: dropping every sender must let
    // `job_rx.blocking_recv()` return `None` immediately and the loop
    // (and thread) exit cleanly.
    drop(job_tx);
    handle
        .join()
        .expect("worker thread must exit and be joinable once every job sender drops");
    assert_eq!(started.load(Ordering::SeqCst), 0);
}

#[test]
fn owner_shutdown_joins_cleanly_once_the_queue_closes() {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));
    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let join_handle = std::thread::spawn(move || {
        run_worker_loop(job_rx, fake_generate(1, started2, ran2));
    });
    let owner = test_owner(join_handle, Duration::from_secs(1));
    drop(job_tx);
    assert_eq!(
        owner._inner.wait_for_exit(Duration::from_secs(1)),
        WorkerShutdown::Joined
    );
    assert_eq!(
        owner._inner.wait_for_exit(Duration::ZERO),
        WorkerShutdown::AlreadyStopped,
        "the join handle must be claimed exactly once"
    );
    assert_eq!(started.load(Ordering::SeqCst), 0);
}

#[test]
fn owner_shutdown_deadline_covers_blocking_thread_local_destructor() {
    let (destructor_entered_tx, destructor_entered_rx) = std::sync::mpsc::sync_channel(1);
    let (release_destructor_tx, release_destructor_rx) = std::sync::mpsc::sync_channel(1);
    let (destructor_finished_tx, destructor_finished_rx) = std::sync::mpsc::sync_channel(1);
    let join_handle = std::thread::spawn(move || {
        BLOCKING_THREAD_LOCAL_DROP.with(|slot| {
            *slot.borrow_mut() = Some(BlockingThreadLocalDrop {
                entered: destructor_entered_tx,
                release: release_destructor_rx,
                finished: destructor_finished_tx,
            });
        });
    });
    destructor_entered_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("the worker must enter its thread-local destructor");
    let finished_deadline = Instant::now() + Duration::from_secs(1);
    while !join_handle.is_finished() && Instant::now() < finished_deadline {
        std::thread::yield_now();
    }
    assert!(
        join_handle.is_finished(),
        "the worker main function must finish while its thread-local destructor is blocked"
    );

    let owner = test_owner(join_handle, Duration::from_millis(20));
    let (shutdown_done_tx, shutdown_done_rx) = std::sync::mpsc::sync_channel(1);
    let shutdown_thread = std::thread::spawn(move || {
        let started = Instant::now();
        let result = owner._inner.wait_for_exit(Duration::from_millis(20));
        let _ = shutdown_done_tx.send((result, started.elapsed()));
    });
    let before_watchdog = shutdown_done_rx
        .recv_timeout(Duration::from_millis(250))
        .ok();

    release_destructor_tx
        .send(())
        .expect("the blocked thread-local destructor must still accept its release");
    destructor_finished_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("the thread-local destructor must finish after release");
    let observed = match before_watchdog {
        Some(result) => result,
        None => shutdown_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("shutdown must finish after the destructor is released"),
    };
    shutdown_thread
        .join()
        .expect("shutdown helper thread must not panic");

    let Some((result, elapsed)) = before_watchdog else {
        panic!(
            "the configured deadline must include thread-local destructor cleanup; \
                 observed {observed:?}"
        );
    };
    assert_eq!(
        result,
        WorkerShutdown::TimedOut,
        "blocked thread-local cleanup must exhaust the configured deadline"
    );
    assert!(
        elapsed < Duration::from_millis(250),
        "shutdown exceeded the deadline watchdog: {elapsed:?}"
    );
}

#[test]
fn final_client_drop_closes_queue_before_owner_joins() {
    let (job_tx, mut job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let (queue_closed_tx, queue_closed_rx) = std::sync::mpsc::sync_channel(1);
    let (allow_exit_tx, allow_exit_rx) = std::sync::mpsc::sync_channel(1);
    let join_handle = std::thread::spawn(move || {
        while job_rx.blocking_recv().is_some() {}
        let _ = queue_closed_tx.send(());
        let _ = allow_exit_rx.recv();
    });
    let owner = test_owner(join_handle, Duration::from_secs(2));
    let client = MetalWorkerClient::with_owner(
        job_tx,
        Arc::new(Semaphore::new(1)),
        Arc::new(AtomicBool::new(false)),
        owner.clone(),
    );
    drop(owner);

    let (drop_done_tx, drop_done_rx) = std::sync::mpsc::sync_channel(1);
    let drop_thread = std::thread::spawn(move || {
        drop(client);
        let _ = drop_done_tx.send(());
    });
    queue_closed_rx
        .recv_timeout(Duration::from_millis(500))
        .expect("dropping the last client must close the queue");
    assert!(
        matches!(
            drop_done_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ),
        "last client drop must still be waiting while the worker is live"
    );
    allow_exit_tx
        .send(())
        .expect("the worker must still be waiting for the exit release");
    drop_done_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("last client drop must finish after the worker exits");
    drop_thread
        .join()
        .expect("client drop thread must not panic");
}

#[test]
fn final_client_drop_timeout_detaches_instead_of_blocking() {
    let (worker_started_tx, worker_started_rx) = std::sync::mpsc::sync_channel(1);
    let (release_worker_tx, release_worker_rx) = std::sync::mpsc::sync_channel(1);
    let (worker_done_tx, worker_done_rx) = std::sync::mpsc::sync_channel(1);
    let join_handle = std::thread::spawn(move || {
        let _ = worker_started_tx.send(());
        let _ = release_worker_rx.recv();
        let _ = worker_done_tx.send(());
    });
    worker_started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("the worker must reach its stuck-backend stand-in");
    let owner = test_owner(join_handle, Duration::from_millis(20));
    let (job_tx, _job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let client = MetalWorkerClient::with_owner(
        job_tx,
        Arc::new(Semaphore::new(1)),
        Arc::new(AtomicBool::new(false)),
        owner.clone(),
    );
    drop(owner);

    let (drop_done_tx, drop_done_rx) = std::sync::mpsc::sync_channel(1);
    let drop_thread = std::thread::spawn(move || {
        drop(client);
        let _ = drop_done_tx.send(());
    });
    let returned_before_watchdog = drop_done_rx
        .recv_timeout(Duration::from_millis(500))
        .is_ok();

    release_worker_tx
        .send(())
        .expect("detached worker must still accept the cleanup release");
    worker_done_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("detached worker must exit after the cleanup release");
    drop_thread
        .join()
        .expect("timed-out client drop thread must not panic");
    assert!(
        returned_before_watchdog,
        "last client Drop must honor its configured deadline instead of joining a stuck worker"
    );
}

#[test]
fn owner_shutdown_reports_worker_panic_after_join() {
    let join_handle = std::thread::spawn(move || {
        panic!("simulated worker panic");
    });
    let owner = test_owner(join_handle, Duration::from_secs(1));

    assert_eq!(
        owner._inner.wait_for_exit(Duration::from_secs(1)),
        WorkerShutdown::Panicked
    );
}

#[test]
fn loader_failure_before_readiness_is_reported_without_touching_a_device() {
    // The `Ok` arm's type (`MetalQwen35State`) is never constructed --
    // this typechecks and runs with zero GPU involvement.
    let result = MetalWorker::spawn(
        || Err("simulated load failure".to_string()),
        DEFAULT_MAX_PENDING_JOBS,
        ResidencyLimits::default(),
    );
    match result {
        Err(StartupError::Load(message)) => {
            assert_eq!(message, "simulated load failure");
        }
        other => panic!("expected StartupError::Load, got {other:?}"),
    }
}

#[test]
fn zero_resident_limits_are_rejected_before_loading() {
    for limits in [
        ResidencyLimits {
            max_adapters: 0,
            max_bytes: 8,
        },
        ResidencyLimits {
            max_adapters: 1,
            max_bytes: 0,
        },
    ] {
        let result = MetalWorker::spawn(
            || panic!("invalid limits must be rejected before loading"),
            DEFAULT_MAX_PENDING_JOBS,
            limits,
        );
        assert!(matches!(
            result,
            Err(StartupError::InvalidResidencyLimits { .. })
        ));
    }
}

#[test]
fn startup_error_display_matches_each_variant() {
    assert_eq!(StartupError::Load("boom".to_string()).to_string(), "boom");
    assert_eq!(
        StartupError::ThreadExited.to_string(),
        "worker thread exited before loading finished"
    );
    assert_eq!(
        StartupError::InvalidMaxPending { max_pending: 0 }.to_string(),
        format!(
            "--max-pending must be between 1 and {} (got 0)",
            Semaphore::MAX_PERMITS
        )
    );
}

// ── #939 max_pending boundary tests ───────────────────────────────────
//
// Validated BEFORE `Semaphore::new` in `MetalWorker::spawn`, so -- like
// `loader_failure_before_readiness_is_reported_without_touching_a_device`
// above -- these never construct a real `MetalQwen35State` and need no
// GPU: an out-of-range `max_pending` returns `Err` before `loader` would
// even be called (a loader that panics if invoked proves that).

#[test]
fn max_pending_zero_is_rejected_before_semaphore_new() {
    let result = MetalWorker::spawn(
        || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> {
            panic!("loader must not run: max_pending=0 must be rejected first")
        },
        0,
        ResidencyLimits::default(),
    );
    match result {
        Err(StartupError::InvalidMaxPending { max_pending: 0 }) => {}
        other => panic!("expected InvalidMaxPending{{max_pending: 0}}, got {other:?}"),
    }
}

#[test]
fn max_pending_above_max_permits_is_rejected_before_semaphore_new() {
    let too_big = Semaphore::MAX_PERMITS + 1;
    let result = MetalWorker::spawn(
        || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> {
            panic!("loader must not run: max_pending above MAX_PERMITS must be rejected first")
        },
        too_big,
        ResidencyLimits::default(),
    );
    match result {
        Err(StartupError::InvalidMaxPending { max_pending }) => {
            assert_eq!(max_pending, too_big);
        }
        other => panic!("expected InvalidMaxPending, got {other:?}"),
    }
}

// ── check_prompt_fits_window, ported from lattice_serve.rs's
//    pre-existing `check_prompt_fits_window` test suite ───────────────

fn cfg_with(max_new_tokens: usize, reasoning_budget: Option<usize>) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens,
        reasoning_budget,
        ..Default::default()
    }
}

#[test]
fn check_prompt_fits_window_rejects_when_prompt_plus_decode_overflows() {
    // model_max_context=8, prompt_len=2, max_new_tokens=7, reasoning_budget=None:
    // 2 (prompt) + 7 (decode) + 1 (delimiter) = 10 > 8 -- must reject.
    let cfg = cfg_with(7, None);
    let err = check_prompt_fits_window(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        8,
        2,
        &cfg,
    )
    .unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "context_length_exceeded");
            assert!(
                message.contains("2 tokens") && message.contains("8-token"),
                "error must name the actual prompt length and window: {message}"
            );
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn lattice_context_boundary_accepts_exact_window_and_rejects_one_past() {
    let cfg = cfg_with(7, None);
    assert!(check_prompt_fits_window(ContextWindowPolicy::PromptAndMaxTokens, 8, 1, &cfg).is_ok());
    assert!(check_prompt_fits_window(ContextWindowPolicy::PromptAndMaxTokens, 8, 2, &cfg).is_err());
}

/// `lattice.rs`'s original `check_context_window` rejects a zero-token
/// prompt independent of the window arithmetic; the policy that
/// reproduces that predicate must too, even when `max_new_tokens`
/// alone fits the window. The delimiter policy never had that
/// conjunct and must keep accepting a zero-length prompt that fits.
#[test]
fn lattice_policy_rejects_zero_token_prompt_even_when_window_fits() {
    let cfg = cfg_with(7, None);
    let err =
        check_prompt_fits_window(ContextWindowPolicy::PromptAndMaxTokens, 8, 0, &cfg).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "context_length_exceeded");
            assert!(
                message.contains("0 tokens"),
                "error must name the zero-length prompt: {message}"
            );
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }

    assert!(
        check_prompt_fits_window(
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            9,
            0,
            &cfg_with(7, None),
        )
        .is_ok()
    );
}

#[test]
fn lattice_serve_context_boundary_accepts_exact_window_and_rejects_one_past() {
    let at_boundary = cfg_with(5, Some(1));
    assert!(
        check_prompt_fits_window(
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            8,
            1,
            &at_boundary,
        )
        .is_ok()
    );

    let one_past = cfg_with(6, Some(1));
    assert!(
        check_prompt_fits_window(
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            8,
            1,
            &one_past,
        )
        .is_err()
    );
}

#[test]
fn check_prompt_fits_window_accepts_ordinary_prompt_unclamped() {
    let cfg = cfg_with(50, None);
    assert!(
        check_prompt_fits_window(
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            4096,
            100,
            &cfg,
        )
        .is_ok()
    );
}

// ── admission cap / backpressure (issue #932) ─────────────────────────

/// Cap enforcement: with `max_pending=2`, job 1 (dequeued immediately,
/// running) plus job 2 (queued behind it) fill the cap; a 3rd submission
/// must be rejected with `ApiError::ServiceUnavailable` before it ever
/// reaches the job channel.
///
/// Mutation-verified by hand (issue #932 implementation): temporarily
/// raising the cap passed to `test_client_and_jobs_with_cap` below from
/// 2 to 3 makes the 3rd submission succeed and this test's
/// `expect_err` panic -- confirming the assertion actually depends on
/// the cap value rather than trivially passing regardless.
#[test]
fn submit_rejects_once_admission_cap_reached() {
    let cap = 2;
    let (client, job_rx) = test_client_and_jobs_with_cap(cap);
    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));
    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let handle =
        std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(2000, started2, ran2)));

    // Job 1: admitted, immediately dequeued (nothing else queued yet),
    // and running fake_generate's 2000-iteration/5ms-per-iteration
    // loop -- long enough to stay in-flight for the rest of this test.
    let (guard1, cancel1) = crate::serve::cancel_pair();
    let rx1 = client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel1,
        )
        .expect("job 1 must be admitted: cap=2, 0 outstanding");
    std::thread::sleep(Duration::from_millis(30));

    // Job 2: admitted (2nd of 2 permits); sits queued behind job 1
    // since the single worker thread is still busy with it.
    let (guard2, cancel2) = crate::serve::cancel_pair();
    let rx2 = client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel2,
        )
        .expect("job 2 must be admitted: cap=2, 1 outstanding");

    // Job 3: cap is now full (job 1 in-flight + job 2 queued == 2 ==
    // cap) -- must be rejected, and must never reach the job channel
    // (no tokenization/model work for a rejected admission).
    let (_guard3, cancel3) = crate::serve::cancel_pair();
    let err = client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel3,
        )
        .expect_err("job 3 must be rejected once the cap is reached");
    match err {
        ApiError::ServiceUnavailable { message } => {
            assert!(
                message.contains("outstanding") || message.contains("pending"),
                "rejection message should explain admission capacity: {message}"
            );
        }
        other => panic!("expected ServiceUnavailable, got {other:?}"),
    }

    // Cleanup: cancel jobs 1 and 2 so fake_generate's should_cancel
    // check stops them quickly, then drain and join.
    drop(guard1);
    drop(guard2);
    drop(rx1);
    drop(rx2);
    drop(client);
    handle.join().expect("worker thread must not panic");
}

/// Slot release on NORMAL completion: a cap=1 client must admit a
/// second job only after the first job's terminal `Complete` event has
/// been delivered and `run_worker_loop` has moved past it (dropping the
/// `WorkerJob`, and with it the admission permit it owns).
#[test]
fn admission_slot_is_released_when_a_job_completes() {
    let cap = 1;
    let (client, job_rx) = test_client_and_jobs_with_cap(cap);
    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));
    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let handle =
        std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(5, started2, ran2)));

    let (_guard1, cancel1) = crate::serve::cancel_pair();
    let mut rx1 = client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel1,
        )
        .expect("job 1 must be admitted");

    // Drain job 1 to its terminal Complete event -- fake_generate(5, ..)
    // runs to completion in ~25ms and is never cancelled.
    let mut completed = false;
    while let Some(ev) = rx1.blocking_recv() {
        if matches!(ev, WorkerEvent::Complete(_)) {
            completed = true;
        }
    }
    assert!(completed, "job 1 must complete normally");

    // The permit `run_worker_loop` held for job 1 is dropped along with
    // `job` at the end of that loop iteration, essentially immediately
    // after the `Complete` send above -- retry briefly rather than
    // assume that has already happened on this exact instruction by the
    // time this (different) thread observes the event.
    let mut admitted = false;
    for _ in 0..50 {
        let (_guard2, cancel2) = crate::serve::cancel_pair();
        match client.submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel2,
        ) {
            Ok(_rx2) => {
                admitted = true;
                break;
            }
            Err(_) => std::thread::sleep(Duration::from_millis(5)),
        }
    }
    assert!(
        admitted,
        "slot must be released once job 1 completes, admitting job 2 at the same cap=1"
    );

    drop(client);
    handle.join().expect("worker thread must not panic");
}

/// THE REGRESSION THIS TEST GUARDS (issue #932): a client-cancelled
/// job that is still sitting in the queue (not yet dequeued) must NOT
/// release its admission slot early -- it is still real, unprocessed
/// work occupying a place in the FIFO queue -- but once the worker
/// actually dequeues it and observes the cancellation (sending exactly
/// one `WorkerEvent::Cancelled`, the existing #832 dequeue-time-cancel
/// contract), its slot MUST be released, same as any other terminal
/// outcome. A permit leaked specifically on this path would let the
/// outstanding-job count only ever grow -- every cancelled queued
/// request would permanently cost one admission slot, eventually
/// wedging admission shut with zero real work outstanding.
#[test]
fn admission_slot_is_released_when_a_queued_job_is_cancelled() {
    let cap = 2;
    let (client, job_rx) = test_client_and_jobs_with_cap(cap);
    let started = Arc::new(AtomicUsize::new(0));
    let ran_tokens = Arc::new(AtomicUsize::new(0));
    let started2 = started.clone();
    let ran2 = ran_tokens.clone();
    let handle =
        std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(2000, started2, ran2)));

    // Job 1: admitted, immediately dequeued, running.
    let (guard1, cancel1) = crate::serve::cancel_pair();
    let rx1 = client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel1,
        )
        .expect("job 1 must be admitted");
    std::thread::sleep(Duration::from_millis(30));

    // Job 2: admitted (2nd of 2 permits), queued behind job 1. Cancel
    // it immediately, client-side, WHILE it is still sitting in the
    // queue, unprocessed.
    let (guard2, cancel2) = crate::serve::cancel_pair();
    let mut rx2 = client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel2,
        )
        .expect("job 2 must be admitted");
    drop(guard2);

    // Cap is full (2/2) right now: a 3rd submit must be rejected --
    // proving a client-cancelled-but-still-queued job legitimately
    // still occupies its slot before it has actually been dequeued.
    let (_guard3, cancel3) = crate::serve::cancel_pair();
    client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel3,
        )
        .expect_err("cap must still be full: job 2's slot isn't released until dequeued");

    // Let job 1 finish (cancel it too) so the worker dequeues job 2
    // next, observes its cancel flag, and emits exactly one Cancelled
    // event for it.
    drop(guard1);
    match rx2.blocking_recv() {
        Some(WorkerEvent::Cancelled) => {}
        other => panic!("expected job 2's exactly-one Cancelled event, got {other:?}"),
    }

    // The slot must now be free -- and specifically BOTH slots, not
    // just one. A single successful 4th admission (the original form
    // of this assertion) does not distinguish "job 2's queued-cancel
    // path correctly released its own permit" from "only job 1's
    // ordinary completion released a permit and job 2's leaked": at
    // this point job 1 has already finished (releasing one permit
    // unconditionally, regression or not), so a leak confined to job
    // 2's queued-cancel path still leaves exactly one usable permit --
    // enough for one admission to spuriously succeed. Poll the
    // semaphore's own count directly (this test module is a child of
    // `metal_worker`, so `client.admission` -- private outside this
    // file -- is visible here) rather than relying on dequeue timing
    // for a second, indirect proof.
    let mut permits_restored = false;
    for _ in 0..50 {
        if client.admission.available_permits() == cap {
            permits_restored = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        permits_restored,
        "both permits (job 1's own release AND job 2's queued-cancel release) must be \
             free once job 2's Cancelled event has fired, got {} of {cap}",
        client.admission.available_permits()
    );

    // And the caller-observable contract still holds: a fresh
    // admission at full cap succeeds.
    let (_guard4, cancel4) = crate::serve::cancel_pair();
    client
        .submit(
            vec![ChatMessage::user("hi")],
            GenerateConfig::default(),
            cancel4,
        )
        .expect("job 2's slot must be released after its Cancelled event, not leaked");

    drop(rx1);
    drop(client);
    handle.join().expect("worker thread must not panic");
}

fn long_text_prompt(chars: usize) -> Vec<ChatMessage> {
    vec![ChatMessage::user("ab".repeat(chars / 2))]
}

fn untruncated_prompt_len(messages: &[ChatMessage]) -> usize {
    tiny_tokenizer()
        .tokenize(&format_chat_template(messages))
        .pre_truncation_len
}

/// Submits one text job to a fake worker built from `tokenizer` and
/// returns its terminal event plus the prompt length handed to generation
/// (`None` when generation was never reached).
fn run_text_job(
    policy: ContextWindowPolicy,
    model_max_context: usize,
    tokenizer: BpeTokenizer,
    messages: Vec<ChatMessage>,
    cfg: GenerateConfig,
) -> (WorkerEvent, Option<usize>) {
    let seen = Arc::new(Mutex::new(None));
    let seen_by_generate = Arc::clone(&seen);
    let client = spawn_fake(
        policy,
        model_max_context,
        tokenizer,
        move |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
            *seen_by_generate.lock().expect("prompt length slot") = Some(prompt_tokens);
            Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens,
                generated_tokens: 0,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            })
        },
    );
    let (_guard, cancel) = crate::serve::cancel_pair();
    let mut rx = client
        .submit(messages, cfg, cancel)
        .expect("the fake worker admits every job");
    let terminal = rx
        .blocking_recv()
        .expect("the job must produce a terminal event");
    let generated_with = *seen.lock().expect("prompt length slot");
    (terminal, generated_with)
}

#[test]
fn thinking_mode_reaches_generation_independent_of_legacy_flag() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_by_generate = Arc::clone(&seen);
    let client = spawn_fake(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        thinking_tokenizer(),
        move |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
            seen_by_generate
                .lock()
                .expect("prompt length slot")
                .push(prompt_tokens);
            Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens,
                generated_tokens: 0,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            })
        },
    );
    let messages = vec![ChatMessage::user("hello")];
    for (enable_thinking, explicitly_thinking) in [(true, true), (true, false), (false, true)] {
        let (_guard, cancel) = crate::serve::cancel_pair();
        let mut rx = client
            .submit_with_lora_mode_and_thinking(
                messages.clone(),
                GenerateConfig {
                    enable_thinking,
                    ..GenerateConfig::default()
                },
                cancel,
                Vec::new(),
                false,
                explicitly_thinking,
            )
            .expect("thinking job is admitted");
        assert!(matches!(rx.blocking_recv(), Some(WorkerEvent::Complete(_))));
    }
    let lengths = seen.lock().expect("prompt length slot");
    assert_eq!(
        lengths[0], lengths[1],
        "the legacy flag must not change the prompt: {lengths:?}"
    );
    assert_ne!(
        lengths[0], lengths[2],
        "the configured thinking mode must change the prompt: {lengths:?}"
    );
}

#[test]
fn text_prompt_above_the_default_tokenizer_cap_reaches_generation_in_full() {
    let messages = long_text_prompt(10_000);
    let full_len = untruncated_prompt_len(&messages);
    let default_cap = crate::tokenizer::bpe::DEFAULT_BPE_MAX_SEQ_LEN;
    let window = 16_384;
    assert_eq!(tiny_tokenizer().max_seq_len(), default_cap);
    assert!(
        full_len > default_cap && full_len < window,
        "fixture must exceed the default cap ({default_cap}) and fit the window ({window}), \
             got {full_len}"
    );
    let cfg = GenerateConfig {
        max_new_tokens: 16,
        ..GenerateConfig::default()
    };

    let (terminal, generated_with) = run_text_job(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        window,
        tiny_tokenizer(),
        messages,
        cfg,
    );
    assert!(
        matches!(terminal, WorkerEvent::Complete(_)),
        "a prompt inside the window must be generated, got {terminal:?}"
    );
    assert_eq!(generated_with, Some(full_len));

    let prompt = format_chat_template(&long_text_prompt(10_000));
    let generation_input = serving_tokenizer(tiny_tokenizer(), window).tokenize(&prompt);
    assert_eq!(
        generation_input.real_length, full_len,
        "the worker's tokenizer must hand generation every prompt token"
    );
}

#[test]
fn text_prompt_beyond_the_window_is_refused_with_its_full_token_count() {
    let window = 64;
    let messages = long_text_prompt(400);
    let full_len = untruncated_prompt_len(&messages);
    let tokenizer = tiny_tokenizer().with_max_seq_len(window / 2);
    assert!(
        full_len > window,
        "fixture must exceed the window, got {full_len}"
    );

    let cfg = GenerateConfig {
        max_new_tokens: 16,
        ..GenerateConfig::default()
    };
    let (terminal, generated_with) = run_text_job(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        window,
        tokenizer.clone(),
        messages.clone(),
        cfg,
    );
    assert_eq!(
        generated_with, None,
        "an over-window prompt must not be generated"
    );
    match terminal {
        WorkerEvent::Rejected(ApiError::BadRequest { message, code }) => {
            assert_eq!(code, "context_length_exceeded");
            assert!(
                message.contains(&format!("prompt has {full_len} tokens")),
                "the refusal must report the untruncated prompt length {full_len}: {message}"
            );
        }
        other => panic!("expected a context-window refusal, got {other:?}"),
    }

    let zero_budget = GenerateConfig {
        max_new_tokens: 0,
        ..GenerateConfig::default()
    };
    let (terminal, generated_with) = run_text_job(
        ContextWindowPolicy::PromptAndMaxTokens,
        window,
        tokenizer,
        messages,
        zero_budget,
    );
    assert_eq!(
        generated_with, None,
        "a prompt longer than the window must be refused even with no decode budget"
    );
    assert!(
        matches!(
            terminal,
            WorkerEvent::Rejected(ApiError::BadRequest {
                code: "context_length_exceeded",
                ..
            })
        ),
        "expected a context-window refusal, got {terminal:?}"
    );
}

mod characterization;
