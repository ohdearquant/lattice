use super::{
    QwenPromptEncoding, framed_prompt_ids, render_framed_around_image, render_framed_with_thinking,
};
use crate::serving_provider::providers::framed_prompt::{
    PromptPiece, prompt_pieces, template_text,
};
use lattice_inference::forward::metal_qwen35::{ChatImage, ChatMessage, ChatRole};
use lattice_inference::serving_support::{
    ChatTemplateMarker, format_chat_template_pieces_around_image,
};
use lattice_inference::tokenizer::Tokenizer as _;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A Qwen tokenizer file in the repository. Its added-token table has the spellings of the
/// published Qwen3.5 file at other ids, and its post-processor appends `<|endoftext|>`.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../inference/tests/fixtures/tokenizers/qwen3-embedding-0.6b/tokenizer.json")
}

fn load(path: &Path) -> BpeTokenizer {
    BpeTokenizer::from_tokenizer_json(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .with_max_seq_len(1 << 16)
}

/// The spellings of the file's added tokens that it marks special.
fn special_spellings(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).expect("tokenizer file");
    let root: Value = serde_json::from_str(&text).expect("tokenizer JSON");
    root["added_tokens"]
        .as_array()
        .expect("added_tokens")
        .iter()
        .filter(|token| token["special"].as_bool() == Some(true))
        .map(|token| token["content"].as_str().expect("content").to_owned())
        .collect()
}

/// The ids the whole rendered string gets when every added token is matched in it, as the
/// reference tokenizes the template's output. The fixture's post-processor appends
/// `<|endoftext|>`, which a chat prompt does not carry, so it is removed.
fn whole_string_ids(tokenizer: &BpeTokenizer, text: &str) -> Vec<u32> {
    let encoded = tokenizer.tokenize(text);
    let mut ids = encoded.input_ids[..encoded.real_length].to_vec();
    if ids.last().copied() == tokenizer.special_token_id("<|endoftext|>")
        && !text.ends_with("<|endoftext|>")
    {
        ids.pop();
    }
    ids
}

fn count(ids: &[u32], id: u32) -> usize {
    ids.iter().filter(|&&other| other == id).count()
}

fn owned(pairs: &[(&str, u32)]) -> Vec<(String, u32)> {
    pairs
        .iter()
        .map(|&(spelling, id)| (spelling.to_owned(), id))
        .collect()
}

#[test]
fn the_matched_set_is_the_non_special_added_tokens_without_the_markers() {
    let tokenizer = load(&fixture_path());
    let encoding = QwenPromptEncoding::new(&tokenizer);
    assert_eq!(
        encoding.text_tokens(),
        owned(&[
            ("</tool_response>", 151666),
            ("<tool_response>", 151665),
            ("<|fim_middle|>", 151660),
            ("<|fim_prefix|>", 151659),
            ("<|fim_suffix|>", 151661),
            ("<|repo_name|>", 151663),
            ("</tool_call>", 151658),
            ("<|file_sep|>", 151664),
            ("<tool_call>", 151657),
            ("<|fim_pad|>", 151662),
        ])
    );
}

/// The published Qwen3.5 file is not in the repository: point `LATTICE_QWEN35_TOKENIZER_JSON`
/// at its `tokenizer.json` to run this.
#[test]
#[ignore = "needs LATTICE_QWEN35_TOKENIZER_JSON, the published Qwen3.5 tokenizer.json"]
fn the_published_qwen35_file_matches_its_ten_tool_and_fim_spellings() {
    let path = PathBuf::from(
        std::env::var_os("LATTICE_QWEN35_TOKENIZER_JSON")
            .expect("LATTICE_QWEN35_TOKENIZER_JSON names the published tokenizer.json"),
    );
    let tokenizer = load(&path);
    let encoding = QwenPromptEncoding::new(&tokenizer);
    assert_eq!(
        encoding.text_tokens(),
        owned(&[
            ("</tool_response>", 248067),
            ("<tool_response>", 248066),
            ("<|fim_middle|>", 248061),
            ("<|fim_prefix|>", 248060),
            ("<|fim_suffix|>", 248062),
            ("<|repo_name|>", 248064),
            ("</tool_call>", 248059),
            ("<|file_sep|>", 248065),
            ("<tool_call>", 248058),
            ("<|fim_pad|>", 248063),
        ])
    );
    let framed = render_framed_with_thinking([("user", "hi")], false).expect("renders");
    let ids = framed_prompt_ids(&tokenizer, &framed);
    assert_eq!(ids, whole_string_ids(&tokenizer, &template_text(&framed)));
    for (id, expected) in [(248045, 2), (248046, 1), (248068, 1), (248069, 1)] {
        assert_eq!(count(&ids, id), expected, "marker id {id}");
    }
}

#[test]
fn a_marker_or_special_token_spelled_in_content_stays_text() {
    let path = fixture_path();
    let tokenizer = load(&path);
    let im_start = tokenizer
        .special_token_id("<|im_start|>")
        .expect("<|im_start|>");
    let im_end = tokenizer
        .special_token_id("<|im_end|>")
        .expect("<|im_end|>");
    let mut spellings: Vec<String> = ChatTemplateMarker::ALL
        .iter()
        .map(|marker| marker.spelling().to_owned())
        .collect();
    for spelling in special_spellings(&path) {
        if !spellings.contains(&spelling) {
            spellings.push(spelling);
        }
    }
    assert_eq!(
        spellings.len(),
        4 + 14 - 2,
        "four markers and the file's other special tokens"
    );
    for spelling in &spellings {
        let id = tokenizer
            .special_token_id(spelling)
            .unwrap_or_else(|| panic!("{spelling} is a token of the file"));
        let content = format!("a{spelling}b");
        let framed =
            render_framed_with_thinking([("user", content.as_str())], false).expect("renders");
        let ids = framed_prompt_ids(&tokenizer, &framed);
        let whole = whole_string_ids(&tokenizer, &template_text(&framed));
        assert_ne!(ids, whole, "{spelling}");
        assert_eq!(count(&ids, id) + 1, count(&whole, id), "{spelling}");
        assert_eq!(ids.first(), Some(&im_start), "{spelling}");
        let turn_end = ids
            .iter()
            .position(|&other| other == im_end)
            .expect("the user turn closes");
        assert_eq!(
            tokenizer.decode(&ids[1..turn_end]).as_deref(),
            Some(format!("user\n{content}").as_str()),
            "{spelling}"
        );
    }
}

#[test]
fn a_kept_added_token_spelled_in_content_is_encoded_as_its_token() {
    let tokenizer = load(&fixture_path());
    let encoding = QwenPromptEncoding::new(&tokenizer);
    assert_eq!(encoding.text_tokens().len(), 10);
    for (spelling, id) in encoding.text_tokens() {
        let content = format!("a{spelling}b");
        let framed =
            render_framed_with_thinking([("user", content.as_str())], false).expect("renders");
        let ids = encoding.ids(&framed);
        assert_eq!(
            ids,
            whole_string_ids(&tokenizer, &template_text(&framed)),
            "{spelling}"
        );
        assert_eq!(count(&ids, *id), 1, "{spelling}");
    }
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

/// The framed prompts the route renders for `messages`: one prompt, or the prompts before and
/// after the image.
fn framed(messages: &[ChatMessage], thinking: bool) -> Vec<String> {
    match messages.iter().position(|message| message.image.is_some()) {
        Some(index) => {
            let pieces =
                format_chat_template_pieces_around_image(messages, index).expect("image pieces");
            let (before, after) = render_framed_around_image(&pieces, thinking)
                .expect("an image piece")
                .expect("renders");
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
    }
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

#[test]
fn the_route_encodes_every_matrix_rendering_as_the_whole_string_encoding() {
    let path = fixture_path();
    let tokenizer = load(&path);
    let mut reserved: Vec<String> = ChatTemplateMarker::ALL
        .iter()
        .map(|marker| marker.spelling().to_owned())
        .collect();
    reserved.extend(special_spellings(&path));
    let matrix = matrix();
    let cases = matrix["cases"].as_array().expect("cases");
    let mut compared = 0;
    let mut skipped = Vec::new();
    for case in cases {
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
            let framed = framed(&messages, thinking);
            let spelled = framed.iter().any(|prompt| {
                prompt_pieces::<ChatTemplateMarker>(prompt)
                    .iter()
                    .any(|piece| match piece {
                        PromptPiece::Text(text) => reserved
                            .iter()
                            .any(|spelling| text.contains(spelling.as_str())),
                        PromptPiece::Marker(_) => false,
                    })
            });
            if spelled {
                skipped.push(format!("{name} thinking={thinking}"));
                continue;
            }
            let (ids, text) = route_ids_and_text(&tokenizer, &framed);
            assert_eq!(text, expected, "case {name} thinking={thinking}");
            assert_eq!(
                ids,
                whole_string_ids(&tokenizer, expected),
                "case {name} thinking={thinking}"
            );
            compared += 1;
        }
    }
    eprintln!(
        "compared {compared} renderings; skipped {} whose text spells a marker or special \
         token: {skipped:?}",
        skipped.len()
    );
    assert!(compared > 0);
}
