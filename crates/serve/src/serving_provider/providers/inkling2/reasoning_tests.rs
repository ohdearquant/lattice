//! The reasoning field through the serving handle. A synthetic Inkling 2 model is served on the
//! CPU host with a tokenizer under which its greedy reply opens a reasoning message, so the
//! completion call and the streaming call run the real decoder, reply split and runtime.

use super::cpu::{Inkling2CpuServing, handle};
use crate::serving_host::{LatticeExecutionRequest, ServingHandle};
use lattice_inference::StopReason;
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_model_inkling2::{Inkling2CpuDecoder, Inkling2Model, synthetic};
use std::sync::{Arc, Mutex};
use tokio::sync::{Semaphore, oneshot, watch};

/// The prompt's token ids. Each is a letter of the fixture tokenizer, never a marker.
const PROMPT: [u32; 11] = [3, 17, 5, 36, 9, 0, 22, 14, 8, 30, 11];

/// The length of the greedy reply the fixtures are cut from.
const STEPS: usize = 16;

const THINKING: &str = "<|content_thinking|>";
const END_MESSAGE: &str = "<|end_message|>";
const TEXT: &str = "<|content_text|>";

/// Every marker that the serving provider or the reply split looks up.
const MARKERS: [&str; 10] = [
    "<|message_system|>",
    "<|message_user|>",
    "<|message_model|>",
    TEXT,
    END_MESSAGE,
    "<|content_model_end_sampling|>",
    THINKING,
    "<|content_tool_error|>",
    "<|content_invoke_tool_text|>",
    "<|content_invoke_tool_json|>",
];

fn letter(id: u32) -> char {
    ('a'..='z')
        .chain('A'..='Z')
        .nth(id as usize)
        .expect("an id with a letter")
}

fn letters(ids: &[u32]) -> String {
    ids.iter().map(|&id| letter(id)).collect()
}

/// `STEPS` ids chosen by hand after [`PROMPT`]: the argmax of each forward pass, fed back one at
/// a time.
fn greedy_ids(model: &Inkling2Model) -> Vec<u32> {
    let mut cache = model.new_cache();
    let mut logits = model.forward(&PROMPT, &mut cache).expect("prompt");
    let mut ids = Vec::new();
    for _ in 0..STEPS {
        let mut best = 0;
        for (index, &value) in logits.iter().enumerate() {
            if value > logits[best] {
                best = index;
            }
        }
        let id = best as u32;
        ids.push(id);
        logits = model.forward(&[id], &mut cache).expect("step");
    }
    ids
}

/// Whether the ids at `positions` can be spelled as markers: they differ, none is a prompt id,
/// and none occurs in `ids` outside `positions`.
fn markable(ids: &[u32], positions: &[usize]) -> bool {
    let marked: Vec<u32> = positions.iter().map(|&position| ids[position]).collect();
    marked
        .iter()
        .enumerate()
        .all(|(index, id)| !marked[..index].contains(id) && !PROMPT.contains(id))
        && ids
            .iter()
            .enumerate()
            .all(|(position, id)| !marked.contains(id) || positions.contains(&position))
}

/// A served synthetic model, its prompt, and the reasoning and content of its greedy reply.
struct Reply {
    handle: ServingHandle,
    prompt: String,
    reasoning: String,
    content: String,
    /// The position of the reply's `<|end_message|>`, when it has a reasoning message.
    close: Option<usize>,
}

/// Serve the synthetic model with each marker of `assigned` spelled by the reply id at its
/// position. The other markers and the end-of-sequence id take ids that neither the prompt nor
/// the reply uses, and every remaining id is its letter.
fn serve(
    model: Inkling2Model,
    ids: &[u32],
    assigned: &[(usize, &'static str)],
) -> (ServingHandle, String) {
    let mut markers: Vec<(u32, &str)> = assigned
        .iter()
        .map(|&(position, name)| (ids[position], name))
        .collect();
    let mut free = (0..37u32).filter(|id| !PROMPT.contains(id) && !ids.contains(id));
    for name in MARKERS {
        if !markers.iter().any(|(_, marker)| *marker == name) {
            markers.push((free.next().expect("a free id"), name));
        }
    }
    let eos = free.next().expect("a free id");
    let mut vocab = Vec::new();
    let mut added = Vec::new();
    for id in 0..37u32 {
        match markers.iter().find(|(marker, _)| *marker == id) {
            Some((_, name)) => added.push(format!(
                r#"{{"id": {id}, "content": "{name}", "special": true}}"#
            )),
            None => vocab.push(format!(r#""{}": {id}"#, letter(id))),
        }
    }
    let json = format!(
        r#"{{"model": {{"type": "BPE", "merges": [], "vocab": {{{}}}}}, "added_tokens": [{}]}}"#,
        vocab.join(", "),
        added.join(", ")
    );
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(&json).expect("fixture tokenizer");
    let decoder = Inkling2CpuDecoder::new(model, tokenizer, eos).expect("decoder");
    let serving = Inkling2CpuServing::new(decoder, eos, 64).expect("serving");
    let prompt = letters(&PROMPT);
    assert_eq!(serving.tokenize_prompt(&prompt), PROMPT);
    let handle = ServingHandle::shared_cpu(handle(serving), Arc::new(Semaphore::new(4)));
    (handle, prompt)
}

/// A reply of a reasoning message and then a content message: `<|content_thinking|>`, the
/// reasoning body, `<|end_message|>`, `<|content_text|>` and the content body.
fn reasoning_then_content() -> Reply {
    let model = synthetic::model(3, 0.2);
    let ids = greedy_ids(&model);
    let close = (2..ids.len() - 2)
        .find(|&close| markable(&ids, &[0, close, close + 1]))
        .unwrap_or_else(|| panic!("no reasoning-then-content cut in {ids:?}"));
    let (handle, prompt) = serve(
        model,
        &ids,
        &[(0, THINKING), (close, END_MESSAGE), (close + 1, TEXT)],
    );
    Reply {
        handle,
        prompt,
        reasoning: letters(&ids[1..close]),
        content: letters(&ids[close + 2..]),
        close: Some(close),
    }
}

/// A reply of one content message and no reasoning.
fn content_only() -> Reply {
    let model = synthetic::model(3, 0.2);
    let ids = greedy_ids(&model);
    assert!(markable(&ids, &[0]), "the first id recurs in {ids:?}");
    let (handle, prompt) = serve(model, &ids, &[(0, TEXT)]);
    Reply {
        handle,
        prompt,
        reasoning: String::new(),
        content: letters(&ids[1..]),
        close: None,
    }
}

fn request(prompt: &str, max_new_tokens: usize) -> LatticeExecutionRequest {
    let mut config = GenerateConfig::default();
    config.max_new_tokens = max_new_tokens;
    config.temperature = 0.0;
    config.repetition_penalty = 1.0;
    config.stop_token_ids = Vec::new();
    config.enable_thinking = false;
    config.seed = Some(7);
    LatticeExecutionRequest {
        prompt: prompt.to_owned(),
        config,
        messages: None,
        selection: Vec::new(),
        explicitly_thinking: false,
    }
}

/// The pieces the streaming call passed to each sink, in arrival order, and its output.
async fn stream(
    reply: &Reply,
    max_new_tokens: usize,
    cancel: watch::Receiver<bool>,
    mut on_reasoning: impl FnMut(&str) -> bool + Send + 'static,
) -> (Vec<(&'static str, String)>, GenerateOutput) {
    let pieces = Arc::new(Mutex::new(Vec::new()));
    let content = Arc::clone(&pieces);
    let reasoning = Arc::clone(&pieces);
    let (finish, finished) = oneshot::channel();
    reply
        .handle
        .stream_lattice(
            request(&reply.prompt, max_new_tokens),
            cancel,
            move |piece| {
                content
                    .lock()
                    .expect("pieces")
                    .push(("content", piece.to_owned()));
                true
            },
            move |piece| {
                reasoning
                    .lock()
                    .expect("pieces")
                    .push(("reasoning", piece.to_owned()));
                on_reasoning(piece)
            },
            move |result| {
                let _ = finish.send(result);
            },
        )
        .await
        .expect("admitted");
    let output = finished
        .await
        .expect("the finish callback runs")
        .expect("generation succeeds");
    let pieces = pieces.lock().expect("pieces").clone();
    (pieces, output)
}

fn joined(pieces: &[(&'static str, String)], field: &str) -> String {
    pieces
        .iter()
        .filter(|(kind, _)| *kind == field)
        .map(|(_, piece)| piece.as_str())
        .collect()
}

#[tokio::test]
async fn the_completion_call_returns_the_reasoning_beside_the_content() {
    let reply = reasoning_then_content();
    let served = reply
        .handle
        .complete_lattice(request(&reply.prompt, STEPS))
        .await
        .expect("completion");
    assert!(!reply.reasoning.is_empty() && !reply.content.is_empty());
    assert_eq!(served.reasoning.as_deref(), Some(reply.reasoning.as_str()));
    assert_eq!(served.output.text, reply.content);
    assert_eq!(served.output.generated_tokens, STEPS);
}

#[tokio::test]
async fn the_streaming_call_passes_the_reasoning_before_the_content_in_generated_order() {
    let reply = reasoning_then_content();
    let (pieces, output) = stream(&reply, STEPS, watch::channel(false).1, |_| true).await;
    let first_content = pieces
        .iter()
        .position(|(kind, _)| *kind == "content")
        .expect("a content piece");
    assert!(first_content > 0, "{pieces:?}");
    assert!(
        pieces[first_content..]
            .iter()
            .all(|(kind, _)| *kind == "content"),
        "{pieces:?}"
    );
    assert_eq!(joined(&pieces, "reasoning"), reply.reasoning);
    assert_eq!(joined(&pieces, "content"), reply.content);
    assert_eq!(output.text, reply.content);
    assert_eq!(output.generated_tokens, STEPS);
}

#[tokio::test]
async fn a_reply_with_no_reasoning_message_has_no_reasoning_on_either_call() {
    let reply = content_only();
    let served = reply
        .handle
        .complete_lattice(request(&reply.prompt, STEPS))
        .await
        .expect("completion");
    assert_eq!(served.reasoning, None);
    assert_eq!(served.output.text, reply.content);
    let (pieces, output) = stream(&reply, STEPS, watch::channel(false).1, |_| true).await;
    assert!(
        pieces.iter().all(|(kind, _)| *kind == "content"),
        "{pieces:?}"
    );
    assert_eq!(output.text, reply.content);
    assert_eq!(reply.reasoning, "");
}

/// A client that goes away while the reply is still reasoning: its stream channel refuses the
/// first reasoning piece, or its disconnect sets the cancel flag. Either way the decoder stops
/// after the opener and the first reasoning token, as it stops for a content piece.
#[tokio::test]
async fn a_client_that_disconnects_during_reasoning_stops_the_decoder() {
    let reply = reasoning_then_content();

    let (sender, receiver) = futures::channel::mpsc::unbounded::<String>();
    drop(receiver);
    let (pieces, closed) = stream(&reply, STEPS, watch::channel(false).1, move |piece| {
        sender.unbounded_send(piece.to_owned()).is_ok()
    })
    .await;

    let (cancel, cancel_rx) = watch::channel(false);
    let (flagged_pieces, flagged) = stream(&reply, STEPS, cancel_rx, move |_| {
        let _ = cancel.send(true);
        true
    })
    .await;

    for (pieces, output) in [(pieces, closed), (flagged_pieces, flagged)] {
        assert_eq!(pieces.len(), 1, "{pieces:?}");
        assert_eq!(pieces[0].0, "reasoning");
        assert_eq!(output.generated_tokens, 2);
        assert_eq!(output.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(output.text, "");
    }
}

/// `max_tokens` counts the reasoning and the content together, and the generated-token count
/// that usage reports includes the reasoning tokens.
#[tokio::test]
async fn the_token_limit_bounds_the_reasoning_and_the_content_together() {
    let reply = reasoning_then_content();
    let close = reply.close.expect("a reasoning message");
    let reasoning: Vec<char> = reply.reasoning.chars().collect();
    let content: Vec<char> = reply.content.chars().collect();
    for (limit, expected_reasoning, expected_content) in [
        (2, reasoning[..1].iter().collect::<String>(), String::new()),
        (
            close + 3,
            reply.reasoning.clone(),
            content[..1].iter().collect::<String>(),
        ),
    ] {
        let served = reply
            .handle
            .complete_lattice(request(&reply.prompt, limit))
            .await
            .expect("completion");
        assert_eq!(served.output.generated_tokens, limit);
        assert_eq!(served.output.stop_reason, Some(StopReason::Length));
        assert_eq!(
            served.reasoning.as_deref(),
            Some(expected_reasoning.as_str())
        );
        assert_eq!(served.output.text, expected_content);
    }
}
