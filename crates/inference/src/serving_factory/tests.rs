use super::{ServingFactory, gemma_parts};
use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::GenerateConfig;
use crate::model::serving_runtime::{RuntimeFactory, ServingRuntime};
use crate::serve::lora::{AdapterIndex, ResidencyLimits};
use crate::serve::metal_worker::{MetalWorker, VisionRuntime, WorkerEvent, WorkerMetadata};
use crate::serve::prepare::PreparationHandle;
use crate::serving_cpu::{GemmaCpuServing, tiny_zero_serving};
use std::sync::{Arc, RwLock};

/// The Gemma worker parts around an already loaded checkpoint: the tiny
/// zero-weight model cannot be loaded from a directory.
struct PreloadedGemma(GemmaCpuServing);

impl RuntimeFactory for PreloadedGemma {
    fn build(
        self: Box<Self>,
        _index: Arc<RwLock<AdapterIndex>>,
        _limits: ResidencyLimits,
    ) -> Result<(Box<dyn ServingRuntime>, WorkerMetadata, PreparationHandle), String> {
        Ok(gemma_parts(self.0))
    }
}

fn greedy(max_new_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens,
        temperature: 0.0,
        repetition_penalty: 1.0,
        stop_token_ids: vec![],
        ..Default::default()
    }
}

#[test]
fn a_gemma_worker_serves_streamed_and_unstreamed_jobs_and_refuses_what_it_cannot() {
    let factory = ServingFactory {
        inner: Box::new(PreloadedGemma(tiny_zero_serving())),
    };
    let (_owner, client, metadata, _preparation) =
        MetalWorker::spawn_with_vision(factory, 4, ResidencyLimits::default())
            .expect("the Gemma worker starts");
    assert_eq!(metadata.format, "safetensors");
    assert!(!client.supports_vision());
    assert!(
        client
            .preparation()
            .is_some_and(|preparation| !preparation.supports_adapters())
    );

    for stream in [false, true] {
        let (_guard, cancel) = crate::serve::cancel_pair();
        let mut rx = client
            .submit_with_lora_mode(
                vec![ChatMessage::user("hello")],
                greedy(3),
                cancel,
                Vec::new(),
                stream,
            )
            .expect("the job is admitted");
        let mut text = String::new();
        let output = loop {
            match rx.blocking_recv().expect("the worker answers") {
                WorkerEvent::Delta(delta) => text.push_str(&delta),
                WorkerEvent::Complete(output) => break output,
                other => panic!("stream={stream}: unexpected event {other:?}"),
            }
        };
        assert_eq!(output.generated_tokens, 3, "stream={stream}");
        assert_eq!(text, output.text, "stream={stream}");
    }

    let (_guard, cancel) = crate::serve::cancel_pair();
    let long = "word ".repeat(metadata.model_max_context + 8);
    let mut rx = client
        .submit_with_lora_mode(
            vec![ChatMessage::user(long)],
            greedy(3),
            cancel,
            Vec::new(),
            false,
        )
        .expect("the job is admitted");
    match rx.blocking_recv().expect("the worker answers") {
        WorkerEvent::Rejected(error) => assert_eq!(error.code(), "context_length_exceeded"),
        other => panic!("expected a window rejection, got {other:?}"),
    }
}

#[test]
fn a_gemma_directory_that_cannot_load_is_reported_by_family() {
    let dir = tempfile::tempdir().expect("temp model dir");
    let factory = ServingFactory::gemma_cpu(dir.path().to_path_buf());
    let error = factory
        .build(
            Arc::new(RwLock::new(AdapterIndex::default())),
            ResidencyLimits::default(),
        )
        .err()
        .expect("an empty directory is not a Gemma 4 checkpoint");
    assert!(error.starts_with("Gemma 4 model load failed: "), "{error}");
}

#[test]
fn qwen_loader_error_is_returned_unchanged() {
    let factory = ServingFactory::qwen_metal(
        || Err("legacy load error".to_string()),
        VisionRuntime::unsupported(),
    );
    let error = factory
        .build(
            Arc::new(RwLock::new(AdapterIndex::default())),
            ResidencyLimits::default(),
        )
        .err()
        .unwrap();
    assert_eq!(error, "legacy load error");
}
