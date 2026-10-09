#![cfg(feature = "serve")]

mod common;

use common::{Binary, Server, require_checkpoint, response_text};
use serde_json::json;
use std::sync::{Arc, Barrier, Mutex, PoisonError};

static CPU_CELL_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn lattice_qwen_cpu_concurrent_greedy_requests_are_isolated_and_stream_disconnect_recovers() {
    let _cell = CPU_CELL_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(model) =
        require_checkpoint(common::QWEN_DIR_ENV, "qwen3.5-0.8b", "cpu-isolation-qwen")
    else {
        return;
    };
    let server = Server::spawn(Binary::Lattice, &model, &[]);
    assert_concurrent_greedy_results(
        &server,
        "Qwen CPU cell",
        16,
        &[
            "Reply with only the word ALPHA.",
            "Reply with only the word BRAVO.",
            "Reply with only the word CHARLIE.",
            "Reply with only the word DELTA.",
        ],
    );

    let mut stream = server.chat_body(json!({
        "messages": [{"role": "user", "content": "Write a long sequence of numbers."}],
        "max_tokens": 64,
        "stream": true
    }));
    stream["stream"] = json!(true);
    assert_eq!(server.disconnect_stream_early(&stream), 200);

    let after_disconnect = server.request(
        "POST",
        "/v1/chat/completions",
        Some(&server.chat_body(json!({
            "messages": [{"role": "user", "content": "Say the word stable."}]
        }))),
    );
    assert_eq!(after_disconnect.status, 200, "{}", after_disconnect.body);
    assert!(
        !response_text(&after_disconnect).starts_with("<missing completion text:"),
        "{}",
        after_disconnect.body
    );
}

#[test]
fn lattice_gemma_cpu_two_concurrent_greedy_requests_are_isolated() {
    let _cell = CPU_CELL_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(model) = require_checkpoint(
        common::GEMMA_DIR_ENV,
        "gemma-4-e2b-it",
        "cpu-isolation-gemma",
    ) else {
        return;
    };
    let server = Server::spawn(Binary::Lattice, &model, &[]);
    assert_concurrent_greedy_results(
        &server,
        "Gemma CPU cell",
        12,
        &[
            "Reply with only the word KIWI.",
            "Reply with only the word TULIP.",
        ],
    );
}

fn assert_concurrent_greedy_results(
    server: &Server,
    cell: &str,
    max_tokens: usize,
    prompts: &[&str],
) {
    let barrier = Arc::new(Barrier::new(prompts.len() + 1));
    let port = server.port();
    let requests = prompts
        .iter()
        .map(|prompt| {
            let body = server.chat_body(json!({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": false
            }));
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                common::request_at_port(port, &body)
            })
        })
        .collect::<Vec<_>>();

    barrier.wait();
    let concurrent = requests
        .into_iter()
        .map(|request| request.join().expect("concurrent request thread joins"))
        .collect::<Vec<_>>();
    let concurrent_text = concurrent
        .iter()
        .enumerate()
        .map(|(index, response)| {
            assert_eq!(
                response.status, 200,
                "concurrent request {index}: {}",
                response.body
            );
            let text = response_text(response);
            assert!(
                !text.starts_with("<missing completion text:"),
                "concurrent request {index}: {}",
                response.body
            );
            text
        })
        .collect::<Vec<_>>();
    let mut solo_texts = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        let body = server.chat_body(json!({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": false
        }));
        let response = server.request("POST", "/v1/chat/completions", Some(&body));
        assert_eq!(
            response.status, 200,
            "solo request {index}: {}",
            response.body
        );
        solo_texts.push(response_text(&response));
    }

    for left in 0..solo_texts.len() {
        for right in (left + 1)..solo_texts.len() {
            assert_ne!(
                solo_texts[left], solo_texts[right],
                "{cell} solo outputs must be pairwise distinct for prompts {:?} and {:?}",
                prompts[left], prompts[right]
            );
        }
    }
    eprintln!("{cell} pairwise-distinct solo outputs: {solo_texts:?}");

    for (index, prompt) in prompts.iter().enumerate() {
        assert_eq!(
            concurrent_text[index], solo_texts[index],
            "{cell} concurrent output differed from the same prompt sent alone: {prompt}"
        );
    }
}
