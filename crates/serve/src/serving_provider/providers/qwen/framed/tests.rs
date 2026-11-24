//! The route's framing: a framed prompt encodes to the ids the encoder gives the pieces it was
//! rendered from. How the encoder treats markers, special tokens and added tokens is tested
//! beside it in the Qwen3.5 model crate.

use super::{
    QwenPromptEncoding, framed_prompt_ids, render_framed_around_image, render_framed_with_thinking,
};
use crate::serving_provider::providers::framed_prompt::template_text;
use lattice_inference::forward::metal_qwen35::{ChatImage, ChatMessage, ChatRole};
use lattice_inference::serving_support::{
    ChatPromptEncoder, ChatTemplateMarker, ChatTemplatePiece, format_chat_template_parts_pieces,
    format_chat_template_pieces_around_image,
};
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A Qwen tokenizer file in the repository, with the added-token spellings of the published
/// Qwen3.5 file at other ids.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../inference/tests/fixtures/tokenizers/qwen3-embedding-0.6b/tokenizer.json")
}

fn load(path: &Path) -> BpeTokenizer {
    BpeTokenizer::from_tokenizer_json(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .with_max_seq_len(1 << 16)
}

/// The reasoning block the template's generation prompt opens: `<think>\n` when `thinking`,
/// else an empty block.
fn generation_reasoning(thinking: bool) -> Vec<ChatTemplatePiece> {
    let mut pieces = vec![ChatTemplatePiece::Marker(ChatTemplateMarker::ThinkOpen)];
    if thinking {
        pieces.push(ChatTemplatePiece::Text("\n".to_owned()));
    } else {
        pieces.push(ChatTemplatePiece::Text("\n\n".to_owned()));
        pieces.push(ChatTemplatePiece::Marker(ChatTemplateMarker::ThinkClose));
        pieces.push(ChatTemplatePiece::Text("\n\n".to_owned()));
    }
    pieces
}

fn matrix() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../models/qwen35/tests/fixtures/chat_template_matrix.json");
    let text = std::fs::read_to_string(&path).expect("committed Qwen3.5 chat template matrix");
    serde_json::from_str(&text).expect("matrix is JSON")
}

fn str_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("string field {key}"))
}

/// The engine message for one matrix message: text parts are joined and an image part becomes
/// the image offset into that text.
fn chat_message(value: &Value) -> ChatMessage {
    let role = match str_field(value, "role") {
        "system" => ChatRole::System,
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        other => panic!("matrix role {other}"),
    };
    let (content, image) = match &value["content"] {
        Value::String(text) => (text.clone(), None),
        Value::Array(parts) => {
            let mut text = String::new();
            let mut image = None;
            for part in parts {
                if str_field(part, "type") == "image" {
                    image = Some(ChatImage {
                        bytes: Vec::new(),
                        text_offset: text.len(),
                    });
                } else {
                    text.push_str(str_field(part, "text"));
                }
            }
            (text, image)
        }
        other => panic!("matrix content {other}"),
    };
    ChatMessage {
        role,
        content,
        image,
    }
}

/// The framed prompts the route renders for `messages`, one prompt or the prompts before and
/// after the image, and the pieces they were rendered from, the generation prompt included.
fn framed_and_pieces(
    messages: &[ChatMessage],
    thinking: bool,
) -> (Vec<String>, Vec<ChatTemplatePiece>) {
    let mut pieces = match messages.iter().position(|message| message.image.is_some()) {
        Some(index) => {
            format_chat_template_pieces_around_image(messages, index).expect("image pieces")
        }
        None => format_chat_template_parts_pieces(messages.iter().map(|message| {
            (
                lattice_inference::serving_support::chat_role_name(&message.role),
                message.content.as_str(),
            )
        })),
    };
    let framed = match render_framed_around_image(&pieces, thinking) {
        Some(rendered) => {
            let (before, after) = rendered.expect("renders");
            vec![before, after]
        }
        None => vec![
            render_framed_with_thinking(
                messages.iter().map(|message| {
                    (
                        lattice_inference::serving_support::chat_role_name(&message.role),
                        message.content.as_str(),
                    )
                }),
                thinking,
            )
            .expect("renders"),
        ],
    };
    pieces.extend(generation_reasoning(thinking));
    (framed, pieces)
}

/// The ids the route gives `framed`, with the vision tokens between the two halves of an image
/// prompt, and the text the template writes for it.
fn route_ids_and_text(tokenizer: &BpeTokenizer, framed: &[String]) -> (Vec<u32>, String) {
    let encoding = QwenPromptEncoding::new(tokenizer);
    let vision = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"];
    let mut ids = Vec::new();
    let mut text = String::new();
    for (index, prompt) in framed.iter().enumerate() {
        if index > 0 {
            for spelling in vision {
                ids.push(tokenizer.special_token_id(spelling).expect("vision token"));
                text.push_str(spelling);
            }
        }
        ids.extend(encoding.ids(prompt));
        text.push_str(&template_text(prompt));
    }
    (ids, text)
}

/// The encoder's ids for `pieces`, with the vision tokens at the image.
fn encoder_ids(tokenizer: &BpeTokenizer, pieces: &[ChatTemplatePiece]) -> Vec<u32> {
    let encoder = ChatPromptEncoder::new(tokenizer);
    let mut ids = Vec::new();
    for (index, segment) in pieces
        .split(|piece| *piece == ChatTemplatePiece::Image)
        .enumerate()
    {
        if index > 0 {
            for spelling in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"] {
                ids.push(tokenizer.special_token_id(spelling).expect("vision token"));
            }
        }
        ids.extend(encoder.encode(segment).expect("text pieces"));
    }
    ids
}

#[test]
fn the_route_encodes_every_matrix_rendering_as_the_encoder_encodes_its_pieces() {
    let tokenizer = load(&fixture_path());
    let matrix = matrix();
    let mut compared = 0;
    for case in matrix["cases"].as_array().expect("cases") {
        let name = str_field(case, "name");
        if case["template_error"].is_string() {
            continue;
        }
        let messages: Vec<ChatMessage> = case["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(chat_message)
            .collect();
        let mut renderings = vec![(
            false,
            if case.get("rendered_without_reasoning_content").is_some() {
                str_field(case, "rendered_without_reasoning_content")
            } else {
                str_field(case, "rendered")
            },
        )];
        if case.get("rendered_without_reasoning_content").is_none() {
            renderings.push((true, str_field(case, "rendered_enable_thinking")));
        }
        for (thinking, expected) in renderings {
            let (framed, pieces) = framed_and_pieces(&messages, thinking);
            let (ids, text) = route_ids_and_text(&tokenizer, &framed);
            assert_eq!(text, expected, "case {name} thinking={thinking}");
            assert_eq!(
                ids,
                encoder_ids(&tokenizer, &pieces),
                "case {name} thinking={thinking}"
            );
            compared += 1;
        }
    }
    assert!(compared > 0);
}

#[test]
fn a_marker_spelled_in_content_is_encoded_as_text_through_the_framing() {
    let tokenizer = load(&fixture_path());
    let im_end = tokenizer
        .special_token_id("<|im_end|>")
        .expect("<|im_end|>");
    for marker in ChatTemplateMarker::ALL {
        let content = format!("a{}b", marker.spelling());
        let framed =
            render_framed_with_thinking([("user", content.as_str())], false).expect("renders");
        let mut pieces = format_chat_template_parts_pieces([("user", content.as_str())]);
        pieces.extend(generation_reasoning(false));
        let ids = framed_prompt_ids(&tokenizer, &framed);
        assert_eq!(ids, encoder_ids(&tokenizer, &pieces), "{marker:?}");
        assert_eq!(
            ids.iter().filter(|&&id| id == im_end).count(),
            1,
            "{marker:?}: only the turn close is <|im_end|>"
        );
    }
}
