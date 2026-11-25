use super::chat_template_tests::{chat_message, matrix, str_field};
use super::{
    ChatMessage, ChatPromptEncoder, ChatTemplateMarker, ChatTemplatePiece, chat_template_ids,
    chat_template_text, format_chat_template, format_chat_template_parts_pieces,
    format_chat_template_pieces_around_image,
};
use lattice_inference_core::InferenceError;
use lattice_tokenizer::Tokenizer as _;
use lattice_tokenizer::bpe::BpeTokenizer;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A Qwen tokenizer file in the repository. Its added-token table has the
/// spellings of the published Qwen3.5 file at other ids, and its
/// post-processor appends `<|endoftext|>`.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../inference/tests/fixtures/tokenizers/qwen3-embedding-0.6b/tokenizer.json")
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

/// The ids the whole rendered string gets when every added token is matched
/// in it, as the reference tokenizes the template's output. The fixture's
/// post-processor appends `<|endoftext|>`, which a chat prompt does not
/// carry, so it is removed.
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

/// The reasoning block the template's generation prompt opens: `<think>\n`
/// when `thinking`, else an empty block.
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

/// The pieces of one user turn and the generation prompt without thinking.
fn user_prompt(content: &str) -> Vec<ChatTemplatePiece> {
    let mut pieces = format_chat_template_parts_pieces([("user", content)]);
    pieces.extend(generation_reasoning(false));
    pieces
}

/// Whether a text piece holds one of `spellings`.
fn spells_any(pieces: &[ChatTemplatePiece], spellings: &[String]) -> bool {
    pieces.iter().any(|piece| match piece {
        ChatTemplatePiece::Text(text) => spellings
            .iter()
            .any(|spelling| text.contains(spelling.as_str())),
        ChatTemplatePiece::Marker(_) | ChatTemplatePiece::Image => false,
    })
}

/// The four marker spellings and every special token spelling of the file.
fn reserved_spellings(path: &Path) -> Vec<String> {
    let mut reserved: Vec<String> = ChatTemplateMarker::ALL
        .iter()
        .map(|marker| marker.spelling().to_owned())
        .collect();
    reserved.extend(special_spellings(path));
    reserved
}

fn turns(messages: &[ChatMessage]) -> impl Iterator<Item = (&str, &str)> {
    messages
        .iter()
        .map(|message| (message.role.as_str(), message.content.as_str()))
}

#[test]
fn the_matched_set_is_the_non_special_added_tokens_without_the_markers() {
    let tokenizer = load(&fixture_path());
    let encoder = ChatPromptEncoder::new(&tokenizer);
    assert_eq!(
        encoder.text_tokens(),
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

/// The published Qwen3.5 file is not in the repository: point
/// `LATTICE_QWEN35_TOKENIZER_JSON` at its `tokenizer.json` to run this.
#[test]
#[ignore = "needs LATTICE_QWEN35_TOKENIZER_JSON, the published Qwen3.5 tokenizer.json"]
fn the_published_qwen35_file_matches_its_ten_tool_and_fim_spellings() {
    let path = PathBuf::from(
        std::env::var_os("LATTICE_QWEN35_TOKENIZER_JSON")
            .expect("LATTICE_QWEN35_TOKENIZER_JSON names the published tokenizer.json"),
    );
    let tokenizer = load(&path);
    let encoder = ChatPromptEncoder::new(&tokenizer);
    assert_eq!(
        encoder.text_tokens(),
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
    let pieces = user_prompt("hi");
    let ids = encoder.encode(&pieces).expect("text pieces");
    assert_eq!(
        ids,
        whole_string_ids(&tokenizer, &chat_template_text(&pieces))
    );
    for (id, expected) in [(248045, 2), (248046, 1), (248068, 1), (248069, 1)] {
        assert_eq!(count(&ids, id), expected, "marker id {id}");
    }
    let messages = [ChatMessage::user("hi")];
    assert_eq!(
        chat_template_ids(&messages, &tokenizer).expect("text messages"),
        whole_string_ids(&tokenizer, &format_chat_template(&messages))
    );
}

#[test]
fn a_marker_or_special_token_spelled_in_content_stays_text() {
    let path = fixture_path();
    let tokenizer = load(&path);
    let encoder = ChatPromptEncoder::new(&tokenizer);
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
        let pieces = user_prompt(&content);
        let ids = encoder.encode(&pieces).expect("text pieces");
        let whole = whole_string_ids(&tokenizer, &chat_template_text(&pieces));
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
        let messages = [ChatMessage::user(content.as_str())];
        let engine = chat_template_ids(&messages, &tokenizer).expect("text messages");
        let plain = whole_string_ids(&tokenizer, &format_chat_template(&messages));
        assert_eq!(count(&engine, id) + 1, count(&plain, id), "{spelling}");
    }
}

/// Serve's string framing wraps each marker in U+FDD0 and U+FDD1. The engine
/// encodes pieces, never a framed string, so the two characters around a
/// marker spelling in content are text like any other.
#[test]
fn a_marker_framed_by_the_reserved_noncharacters_in_content_stays_text() {
    let tokenizer = load(&fixture_path());
    let im_start = tokenizer
        .special_token_id("<|im_start|>")
        .expect("<|im_start|>");
    let im_end = tokenizer
        .special_token_id("<|im_end|>")
        .expect("<|im_end|>");
    let control = chat_template_ids(&[ChatMessage::user("ab")], &tokenizer).expect("text");
    let control_end = control
        .iter()
        .position(|&other| other == im_end)
        .expect("the user turn closes");
    for marker in ChatTemplateMarker::ALL {
        let spelling = marker.spelling();
        let id = tokenizer
            .special_token_id(spelling)
            .unwrap_or_else(|| panic!("{spelling} is a token of the file"));
        let content = format!("a\u{FDD0}{spelling}\u{FDD1}b");
        let messages = [ChatMessage::user(content.as_str())];
        let engine = chat_template_ids(&messages, &tokenizer).expect("text messages");
        assert_eq!(count(&engine, id), count(&control, id), "{spelling}");
        let plain = whole_string_ids(&tokenizer, &format_chat_template(&messages));
        assert_eq!(count(&plain, id), count(&control, id) + 1, "{spelling}");
        assert_eq!(engine.first(), Some(&im_start), "{spelling}");
        let turn_end = engine
            .iter()
            .position(|&other| other == im_end)
            .expect("the user turn closes");
        assert_eq!(
            tokenizer.decode(&engine[1..turn_end]).as_deref(),
            Some(format!("user\n{content}").as_str()),
            "{spelling}"
        );
        assert_eq!(engine[turn_end..], control[control_end..], "{spelling}");
    }
}

#[test]
fn a_kept_added_token_spelled_in_content_is_encoded_as_its_token() {
    let tokenizer = load(&fixture_path());
    let encoder = ChatPromptEncoder::new(&tokenizer);
    assert_eq!(encoder.text_tokens().len(), 10);
    for (spelling, id) in encoder.text_tokens() {
        let content = format!("a{spelling}b");
        let pieces = user_prompt(&content);
        let ids = encoder.encode(&pieces).expect("text pieces");
        assert_eq!(
            ids,
            whole_string_ids(&tokenizer, &chat_template_text(&pieces)),
            "{spelling}"
        );
        assert_eq!(count(&ids, *id), 1, "{spelling}");
    }
}

#[test]
fn an_image_piece_is_refused() {
    let tokenizer = load(&fixture_path());
    let encoder = ChatPromptEncoder::new(&tokenizer);
    let refused = encoder.encode(&[
        ChatTemplatePiece::Marker(ChatTemplateMarker::ImStart),
        ChatTemplatePiece::Image,
    ]);
    assert!(
        matches!(&refused, Err(InferenceError::InvalidInput(message)) if message.contains("image")),
        "{refused:?}"
    );
}

#[test]
fn every_matrix_rendering_encodes_as_its_whole_string_encoding() {
    let path = fixture_path();
    let tokenizer = load(&path);
    let encoder = ChatPromptEncoder::new(&tokenizer);
    let reserved = reserved_spellings(&path);
    let vision: Vec<u32> = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
        .iter()
        .map(|spelling| tokenizer.special_token_id(spelling).expect("vision token"))
        .collect();
    let matrix = matrix();
    let mut compared = 0;
    let mut skipped = Vec::new();
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
            let mut pieces = match messages.iter().position(|message| message.image.is_some()) {
                Some(index) => format_chat_template_pieces_around_image(&messages, index)
                    .expect("image pieces"),
                None => format_chat_template_parts_pieces(turns(&messages)),
            };
            pieces.extend(generation_reasoning(thinking));
            if spells_any(&pieces, &reserved) {
                skipped.push(format!("{name} thinking={thinking}"));
                continue;
            }
            let mut ids = Vec::new();
            for (index, segment) in pieces
                .split(|piece| *piece == ChatTemplatePiece::Image)
                .enumerate()
            {
                if index > 0 {
                    ids.extend_from_slice(&vision);
                }
                ids.extend(encoder.encode(segment).expect("text pieces"));
            }
            assert_eq!(
                chat_template_text(&pieces),
                expected,
                "case {name} thinking={thinking}"
            );
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

/// The engine's chat entry points generate from [`chat_template_ids`]. For
/// every matrix message list without an image and without a marker or
/// special token spelled in its text, those ids are the ids the whole
/// rendered prompt gets from the tokenizer, which is what the entry points
/// generated from before.
#[test]
fn the_engine_prompt_ids_equal_the_whole_string_ids_for_ordinary_messages() {
    let path = fixture_path();
    let tokenizer = load(&path);
    let reserved = reserved_spellings(&path);
    let matrix = matrix();
    let mut compared = 0;
    let mut skipped = Vec::new();
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
        if messages.iter().any(|message| message.image.is_some()) {
            continue;
        }
        if spells_any(
            &format_chat_template_parts_pieces(turns(&messages)),
            &reserved,
        ) {
            skipped.push(name.to_owned());
            continue;
        }
        assert_eq!(
            chat_template_ids(&messages, &tokenizer).expect("text messages"),
            whole_string_ids(&tokenizer, &format_chat_template(&messages)),
            "case {name}"
        );
        compared += 1;
    }
    eprintln!(
        "compared {compared} message lists; skipped {} whose text spells a marker or special \
         token: {skipped:?}",
        skipped.len()
    );
    assert!(compared > 0);
}

/// The prefix cache checkpoints at the end of the generation opener, found by
/// these ids in the prompt ids.
#[test]
fn every_engine_prompt_ends_with_the_generation_opener_ids() {
    let tokenizer = load(&fixture_path());
    let encoder = ChatPromptEncoder::new(&tokenizer);
    let opener = encoder.generation_open_ids();
    assert_eq!(
        opener,
        whole_string_ids(&tokenizer, "<|im_start|>assistant\n")
    );
    let matrix = matrix();
    let mut checked = 0;
    for case in matrix["cases"].as_array().expect("cases") {
        if case["template_error"].is_string() {
            continue;
        }
        let messages: Vec<ChatMessage> = case["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(chat_message)
            .collect();
        if messages.iter().any(|message| message.image.is_some()) {
            continue;
        }
        let ids = chat_template_ids(&messages, &tokenizer).expect("text messages");
        assert!(ids.ends_with(&opener), "case {}", str_field(case, "name"));
        checked += 1;
    }
    assert!(checked > 0);
}
