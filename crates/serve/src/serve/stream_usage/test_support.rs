//! Shared stream-usage cases for the two HTTP binary test suites.

use lattice_inference::{GenerateOutput, StopReason};
use serde_json::{Value, json};

/// The completion outcomes exercised by each real-router fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    Stop,
    Length,
    Interrupt,
}

/// All supported request flag settings.
pub const OPTIONS: [Option<bool>; 3] = [None, Some(false), Some(true)];
/// All backend completion outcomes in the usage matrix.
pub const COMPLETIONS: [Completion; 3] =
    [Completion::Stop, Completion::Length, Completion::Interrupt];

/// Build one deterministic backend result with authoritative token counts.
pub fn output(completion: Completion) -> GenerateOutput {
    let (stopped, reason) = match completion {
        Completion::Stop => (true, StopReason::Eos),
        Completion::Length => (false, StopReason::Length),
        Completion::Interrupt => (false, StopReason::Interrupt),
    };
    GenerateOutput {
        text: "é".into(),
        token_ids: vec![1, 2],
        prompt_tokens: 3,
        generated_tokens: 2,
        stopped,
        stop_reason: Some(reason),
        token_logprobs: vec![],
    }
}

/// Assert the usage member of an ordinary chunk before stream completion.
pub fn assert_regular_usage(chunk: &Value, option: Option<bool>) {
    if option == Some(true) {
        assert_eq!(chunk.get("usage"), Some(&Value::Null));
    } else {
        assert!(chunk.get("usage").is_none());
    }
}

/// Assert complete SSE frames and authoritative usage for one matrix cell.
pub fn assert_frames(text: &str, option: Option<bool>, completion: Completion) {
    let frames = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    let has_final_usage = option == Some(true) && completion != Completion::Interrupt;
    assert_eq!(frames.len(), if has_final_usage { 5 } else { 4 }, "{text}");
    assert_eq!(frames.last(), Some(&"[DONE]"));
    let chunks = frames[..frames.len() - 1]
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "é");
    assert_eq!(
        chunks[2]["choices"][0]["finish_reason"],
        if completion == Completion::Stop {
            "stop"
        } else {
            "length"
        }
    );
    for chunk in &chunks[..3] {
        assert_regular_usage(chunk, option);
    }
    if has_final_usage {
        assert_eq!(chunks[3]["choices"], json!([]));
        assert_eq!(
            chunks[3]["usage"],
            json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":5})
        );
        assert_eq!(chunks[3]["id"], chunks[0]["id"]);
        assert_eq!(chunks[3]["model"], chunks[0]["model"]);
    }
}
