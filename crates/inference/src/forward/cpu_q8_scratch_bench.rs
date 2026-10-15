use super::{make_nonzero_q8_generate_fixture, nonzero_fixture_tokenizer_q8};
use crate::decoder::standalone_cpu::{StandaloneWeights, generate_with_trace};
use crate::forward::cpu_q8::generate_q8;
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::stop_reason::StopReason;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn checked_checksum(
    output: &GenerateOutput,
    prompt_tokens: usize,
    generated_tokens: usize,
    vocab_size: usize,
) -> u64 {
    assert_eq!(output.prompt_tokens, prompt_tokens);
    assert_eq!(output.generated_tokens, generated_tokens);
    assert_eq!(output.token_ids.len(), generated_tokens);
    assert!(!output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Length));
    assert!(!output.text.is_empty());
    assert!(
        output
            .token_ids
            .iter()
            .all(|&id| (id as usize) < vocab_size)
    );
    assert!(output.token_ids.iter().any(|&id| id != 0));
    let checksum = output
        .token_ids
        .iter()
        .map(|&id| u64::from(id))
        .chain(output.text.bytes().map(u64::from))
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ value).wrapping_mul(0x0000_0100_0000_01b3)
        });
    assert_ne!(checksum, 0);
    checksum
}

#[test]
#[ignore = "manual synthetic CPU request timing; run under benchmark admission"]
fn bench_generate_q8_request_scratch() {
    const ITERATIONS: usize = 5;
    let (cfg, weights, rope) = make_nonzero_q8_generate_fixture();
    let tokenizer = nonzero_fixture_tokenizer_q8();
    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.intermediate_size, 128);
    assert_eq!(cfg.num_hidden_layers, 2);
    assert_eq!(cfg.num_linear_attention_layers(), 1);
    assert_eq!(cfg.num_full_attention_layers(), 1);

    for (label, prompt_repeats, new_tokens) in [
        ("short", 1, 6),
        ("prefill_40", 8, 16),
        ("prefill_160", 32, 16),
    ] {
        let prompt = "world".repeat(prompt_repeats);
        let prompt_tokens = 5 * prompt_repeats;
        let gen_cfg = GenerateConfig {
            max_new_tokens: new_tokens,
            temperature: 0.0,
            seed: Some(7),
            ..Default::default()
        };
        assert!(prompt_tokens + new_tokens < cfg.max_position_embeddings);

        // The trace control is outside the timer; timed calls use the public wrapper.
        let (traced, trace) = generate_with_trace(
            StandaloneWeights::Q8(&weights),
            &cfg,
            &tokenizer,
            &rope,
            &prompt,
            &gen_cfg,
        )
        .expect("synthetic Q8 trace control must succeed");
        let checksum = checked_checksum(&traced, prompt_tokens, new_tokens, cfg.vocab_size);
        assert_eq!(trace.opened, new_tokens);
        assert_eq!(trace.consumed, new_tokens - 1);
        if label == "short" {
            assert_eq!(traced.token_ids, [36, 80, 112, 62, 103, 86]);
            assert_eq!(traced.text, "aCbkbQa2bHbq");
        }

        let warm = generate_q8(&weights, &cfg, &tokenizer, &rope, &prompt, &gen_cfg)
            .expect("synthetic Q8 request warmup must succeed");
        assert_eq!(
            checked_checksum(&warm, prompt_tokens, new_tokens, cfg.vocab_size),
            checksum
        );
        assert_eq!(warm.token_ids, traced.token_ids);
        assert_eq!(warm.text, traced.text);
        black_box(&warm);

        let mut elapsed = Duration::ZERO;
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            let output = generate_q8(
                black_box(&weights),
                black_box(&cfg),
                black_box(&tokenizer),
                black_box(&rope),
                black_box(&prompt),
                black_box(&gen_cfg),
            )
            .expect("timed synthetic Q8 request must succeed");
            elapsed += start.elapsed();
            black_box(&output);
            assert_eq!(
                checked_checksum(&output, prompt_tokens, new_tokens, cfg.vocab_size),
                checksum
            );
            assert_eq!(output.token_ids, traced.token_ids);
            assert_eq!(output.text, traced.text);
        }
        assert!(!elapsed.is_zero());
        eprintln!(
            "q8_request_scratch case={label} hidden={} intermediate={} layers={} prompt_tokens={prompt_tokens} generated_tokens={new_tokens} iterations={ITERATIONS} elapsed_ns={} ns_per_request={:.0} checksum={checksum:016x} opened={} consumed={}",
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
            elapsed.as_nanos(),
            elapsed.as_nanos() as f64 / ITERATIONS as f64,
            trace.opened,
            trace.consumed,
        );
    }
}
