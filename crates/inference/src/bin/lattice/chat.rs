// ---------------------------------------------------------------------------
// chat subcommand
// ---------------------------------------------------------------------------

use crate::backend;

/// Load `config.json` for a Q4 directory, via the single shared
/// config-resolution policy (`Qwen35Config::from_model_dir`, #923) used by
/// every loader in this crate: a missing `config.json` is a hard,
/// descriptive error naming the directory, never a silently-substituted
/// architecture preset.
#[cfg(feature = "metal-gpu")]
pub(crate) fn load_q4_config(
    dir: &std::path::Path,
) -> Result<lattice_inference::model::qwen35_config::Qwen35Config, String> {
    lattice_inference::model::qwen35_config::Qwen35Config::from_model_dir(dir)
        .map_err(|e| format!("config.json load failed: {e}"))
}

/// Metal-GPU chat backend: owns a `MetalQwen35State` plus the tokenizer and
/// context-window cap needed to serve `generate`/`generate_streaming` calls
/// the same way the CPU (`Qwen35Model`) backend does.
///
/// `MetalQwen35State` is `!Send` (it owns raw `metal::*` FFI objects), so
/// this type must never be shared across threads. `run_chat`'s REPL uses it
/// directly on the calling thread; the `serve` module never constructs one
/// on an async task — it lives on a dedicated worker thread instead (see
/// `serve::spawn_metal_worker`).
#[cfg(feature = "metal-gpu")]
struct MetalChatBackend {
    state: lattice_inference::forward::metal_qwen35::MetalQwen35State,
    tokenizer: lattice_inference::tokenizer::bpe::BpeTokenizer,
}

#[cfg(feature = "metal-gpu")]
impl MetalChatBackend {
    /// `max_cache_len` bounds the KV cache (and therefore the usable context
    /// window). 4096 matches the cap used by `chat_metal.rs`.
    const MAX_CACHE_LEN: usize = 4096;

    /// `tokenizer_dir` overrides where `tokenizer.json` is read from, for Q4
    /// directories that were produced without a co-located tokenizer. `None`
    /// resolves it from `dir` itself (the common case: Q4 dirs ship it).
    fn load(
        dir: &std::path::Path,
        tokenizer_dir: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let tokenizer_path = tokenizer_dir.unwrap_or(dir).join("tokenizer.json");
        let tokenizer =
            lattice_inference::tokenizer::bpe::BpeTokenizer::from_tokenizer_json(&tokenizer_path)
                .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?;
        let cfg = load_q4_config(dir)?;
        let state = lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir(
            dir,
            &tokenizer_path,
            &cfg,
            Self::MAX_CACHE_LEN,
        )
        .map_err(|e| format!("Q4 model load failed: {e}"))?;
        Ok(Self { state, tokenizer })
    }

    fn generate(
        &mut self,
        prompt: &str,
        gen_cfg: &lattice_inference::GenerateConfig,
    ) -> Result<lattice_inference::GenerateOutput, lattice_inference::error::InferenceError> {
        let Self { state, tokenizer } = self;
        generate_checked(tokenizer, state.max_context(), prompt, || {
            state.generate(prompt, tokenizer, gen_cfg)
        })
    }
}

/// Narrow accessor for `serve.rs`: the Metal worker needs this cache-length
/// cap without reaching into `MetalChatBackend`'s internals across the
/// module boundary.
#[cfg(feature = "metal-gpu")]
pub(crate) fn chat_max_cache_len() -> usize {
    MetalChatBackend::MAX_CACHE_LEN
}

enum Backend {
    Cpu(Box<lattice_inference::model::qwen35::Qwen35Model>),
    #[cfg(feature = "metal-gpu")]
    Metal(Box<MetalChatBackend>),
}

impl Backend {
    fn cpu(mut model: lattice_inference::model::qwen35::Qwen35Model) -> Self {
        model.ensure_tokenizer_max_seq_len(model.max_context());
        Self::Cpu(Box::new(model))
    }

    fn generate_chat_line(
        &mut self,
        prompt: &str,
        gen_cfg: &lattice_inference::GenerateConfig,
    ) -> Result<lattice_inference::GenerateOutput, lattice_inference::InferenceError> {
        match self {
            Self::Cpu(model) => {
                generate_checked(model.tokenizer(), model.max_context(), prompt, || {
                    model.generate(prompt, gen_cfg)
                })
            }
            #[cfg(feature = "metal-gpu")]
            Self::Metal(model) => model.generate(prompt, gen_cfg),
        }
    }
}

fn generate_checked(
    tokenizer: &lattice_inference::BpeTokenizer,
    limit: usize,
    prompt: &str,
    generate: impl FnOnce() -> Result<
        lattice_inference::GenerateOutput,
        lattice_inference::InferenceError,
    >,
) -> Result<lattice_inference::GenerateOutput, lattice_inference::InferenceError> {
    use lattice_inference::Tokenizer;

    let prompt_tokens = tokenizer.tokenize(prompt).pre_truncation_len;
    if prompt_tokens > limit {
        return Err(lattice_inference::InferenceError::InvalidInput(format!(
            "prompt ({prompt_tokens} tokens) exceeds model context window ({limit})"
        )));
    }
    generate()
}

#[allow(clippy::field_reassign_with_default)]
pub(crate) fn run_chat(
    model_path: &str,
    max_tokens: usize,
    temperature: f32,
    tokenizer_dir: Option<&str>,
    prompt: Option<&str>,
) {
    use std::path::Path;

    let path = Path::new(model_path);
    let format = backend::detect_format(path);
    #[cfg(feature = "metal-gpu")]
    let tokenizer_dir_path = tokenizer_dir.map(Path::new);
    #[cfg(not(feature = "metal-gpu"))]
    let _ = tokenizer_dir;

    eprintln!("Loading model from {model_path}...");

    let mut model = match format {
        backend::ModelFormat::Safetensors => {
            match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(path) {
                Ok(m) => Backend::cpu(m),
                Err(e) => {
                    eprintln!("Error: failed to load model: {e}");
                    std::process::exit(1);
                }
            }
        }
        backend::ModelFormat::Q4 => {
            #[cfg(feature = "metal-gpu")]
            {
                match MetalChatBackend::load(path, tokenizer_dir_path) {
                    Ok(m) => Backend::Metal(Box::new(m)),
                    Err(e) => {
                        eprintln!("Error: failed to load Q4 model: {e}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(feature = "metal-gpu"))]
            {
                eprintln!("Error: {}", backend::metal_gpu_required_message(path));
                std::process::exit(1);
            }
        }
        backend::ModelFormat::Unknown => {
            eprintln!("Error: {}", backend::unrecognized_format_message(path));
            std::process::exit(1);
        }
        // Any format this binary doesn't yet know how to load is handled
        // the same way as `Unknown`: report it and exit, rather than
        // silently guessing a loader.
        _ => {
            eprintln!("Error: {}", backend::unrecognized_format_message(path));
            std::process::exit(1);
        }
    };
    let mut gen_cfg = lattice_inference::GenerateConfig::default();
    gen_cfg.max_new_tokens = max_tokens;
    gen_cfg.temperature = temperature;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    if let Err(error) = run_chat_io(prompt, stdin.lock(), &mut stdout, &mut stderr, |text| {
        model.generate_chat_line(text, &gen_cfg)
    }) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_chat_io(
    prompt: Option<&str>,
    input: impl std::io::BufRead,
    stdout: &mut impl std::io::Write,
    stderr: &mut impl std::io::Write,
    mut generate: impl FnMut(
        &str,
    ) -> Result<
        lattice_inference::GenerateOutput,
        lattice_inference::InferenceError,
    >,
) -> Result<(), String> {
    if let Some(prompt) = prompt {
        let output = generate(prompt).map_err(|e| format!("Generation error: {e}"))?;
        writeln!(stdout, "{}", output.text).map_err(|e| format!("Error writing output: {e}"))?;
        stdout
            .flush()
            .map_err(|e| format!("Error flushing output: {e}"))?;
        return Ok(());
    }

    writeln!(stderr, "Model loaded. Type 'exit' or 'quit' to stop.\n")
        .map_err(|e| format!("Error writing diagnostics: {e}"))?;

    for line in input.lines() {
        let prompt = match line {
            Ok(l) => l,
            Err(e) => {
                writeln!(stderr, "Error reading input: {e}")
                    .map_err(|e| format!("Error writing diagnostics: {e}"))?;
                break;
            }
        };
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            break;
        }

        match generate(trimmed) {
            Ok(output) => {
                let _ = writeln!(stdout, "{}", output.text);
                let _ = writeln!(
                    stdout,
                    "[{} prompt tokens, {} generated]",
                    output.prompt_tokens, output.generated_tokens
                );
            }
            Err(e) => {
                writeln!(stderr, "Generation error: {e}")
                    .map_err(|e| format!("Error writing diagnostics: {e}"))?;
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context;
    use lattice_inference::{GenerateConfig, InferenceError};

    fn count_only() -> GenerateConfig {
        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = 0;
        cfg
    }

    #[test]
    fn repl_uses_checked_generation_and_cpu_initialization() {
        // The stdin-driven entry point must use the same seams as these tests.
        let source = include_str!("chat.rs")
            .split("#[cfg(all(test,")
            .next()
            .unwrap();
        assert!(source.contains("Ok(m) => Backend::cpu(m),"));
        assert!(source.contains("if let Err(error) = run_chat_io("));
        assert!(source.contains("model.generate_chat_line(text, &gen_cfg)"));
        assert!(source.contains("eprintln!(\"{error}\");\n        std::process::exit(1);"));
    }

    #[test]
    fn cpu_chat_generation_keeps_long_prompts() {
        let model = tiny_zero_model_with_context(8192);
        assert_eq!(model.tokenizer().max_seq_len(), 4096);
        let mut backend = Backend::cpu(model);
        for n in [4097, 8192] {
            let output = backend
                .generate_chat_line(&"a".repeat(n), &count_only())
                .unwrap();
            assert_eq!(output.prompt_tokens, n);
            assert_eq!(output.generated_tokens, 0);
        }
    }

    fn assert_refused(backend: &mut Backend, n: usize, limit: usize) {
        let error = backend
            .generate_chat_line(&"a".repeat(n), &count_only())
            .unwrap_err();
        assert!(
            matches!(error, InferenceError::InvalidInput(ref message)
            if message == &format!("prompt ({n} tokens) exceeds model context window ({limit})")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn cpu_chat_refuses_full_count_and_accepts_next_line() {
        for limit in [32, 8192] {
            let mut backend = Backend::cpu(tiny_zero_model_with_context(limit));
            for n in [limit + 1, limit + 137] {
                assert_refused(&mut backend, n, limit);
            }
            let output = backend.generate_chat_line("a", &count_only()).unwrap();
            assert_eq!(output.prompt_tokens, 1);
        }
    }

    #[test]
    fn chat_guard_refuses_before_generation_with_metal_tokenizer_cap() {
        let model = tiny_zero_model_with_context(8192);
        let tokenizer = model.tokenizer();
        let limit = 4096;
        assert_eq!(tokenizer.max_seq_len(), limit);
        for n in [limit + 1, limit + 137] {
            let prompt = "a".repeat(n);
            let mut called = false;
            let result = generate_checked(tokenizer, limit, &prompt, || {
                called = true;
                model.generate(&prompt, &count_only())
            });
            assert!(!called, "an overlong prompt must not reach generation");
            assert!(matches!(result, Err(InferenceError::InvalidInput(message))
                if message == format!("prompt ({n} tokens) exceeds model context window ({limit})")));
        }
        for n in [1, limit] {
            let prompt = "a".repeat(n);
            let output = generate_checked(tokenizer, limit, &prompt, || {
                model.generate(&prompt, &count_only())
            })
            .unwrap();
            assert_eq!(output.prompt_tokens, n);
        }
    }

    #[cfg(feature = "metal-gpu")]
    #[test]
    fn metal_chat_uses_checked_generation() {
        // Metal's compatible fixtures are private to its library tests.
        let source = include_str!("chat.rs")
            .split("enum Backend {")
            .next()
            .unwrap();
        assert!(source.contains("generate_checked(tokenizer, state.max_context(), prompt, ||"));
        assert!(source.contains("state.generate(prompt, tokenizer, gen_cfg)"));
        assert_eq!(MetalChatBackend::MAX_CACHE_LEN, 4096);
    }
}

#[cfg(test)]
mod io_tests {
    use super::run_chat_io;
    use lattice_inference::{GenerateOutput, InferenceError};
    use std::io::{self, BufRead, Cursor, Read, Write};

    struct NoInput;

    impl Read for NoInput {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            panic!("one-shot chat must not read stdin");
        }
    }

    impl BufRead for NoInput {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            panic!("one-shot chat must not buffer stdin");
        }

        fn consume(&mut self, _amount: usize) {
            panic!("one-shot chat must not consume stdin");
        }
    }

    fn output(text: &str) -> GenerateOutput {
        GenerateOutput {
            text: text.into(),
            token_ids: vec![],
            prompt_tokens: 4,
            generated_tokens: 2,
            stopped: false,
            stop_reason: None,
            token_logprobs: vec![],
        }
    }

    #[test]
    fn oneshot_uses_literal_prompt_once_without_reading_stdin() {
        for prompt in ["  literal\nline  ", "exit", "QUIT", "", " \t\n "] {
            let mut calls = 0;
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            run_chat_io(Some(prompt), NoInput, &mut stdout, &mut stderr, |text| {
                calls += 1;
                assert_eq!(text, prompt);
                Ok(output("  generated\ntext  "))
            })
            .unwrap();
            assert_eq!(calls, 1, "prompt {prompt:?}");
            assert_eq!(stdout, b"  generated\ntext  \n");
            assert!(
                stderr.is_empty(),
                "one-shot chat must not print a REPL banner"
            );
        }
    }

    #[test]
    fn oneshot_generation_failure_returns_error_without_output() {
        let mut calls = 0;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let error = run_chat_io(Some("prompt"), NoInput, &mut stdout, &mut stderr, |_| {
            calls += 1;
            Err(InferenceError::InvalidInput("generation refused".into()))
        })
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(error.starts_with("Generation error:"), "{error}");
        assert!(error.contains("generation refused"), "{error}");
        assert!(stdout.is_empty());
        assert!(stderr.is_empty());
    }

    struct FailingOutput {
        fail_write: bool,
        written: Vec<u8>,
    }

    impl Write for FailingOutput {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if self.fail_write {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "write failed"));
            }
            self.written.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "flush failed"))
        }
    }

    #[test]
    fn oneshot_output_failures_are_reported() {
        for fail_write in [true, false] {
            let mut stdout = FailingOutput {
                fail_write,
                written: Vec::new(),
            };
            let mut stderr = Vec::new();
            let mut calls = 0;
            let error = run_chat_io(Some("prompt"), NoInput, &mut stdout, &mut stderr, |_| {
                calls += 1;
                Ok(output("generated"))
            })
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(
                error,
                if fail_write {
                    "Error writing output: write failed"
                } else {
                    "Error flushing output: flush failed"
                }
            );
            assert_eq!(
                stdout.written,
                if fail_write {
                    &b""[..]
                } else {
                    &b"generated\n"[..]
                }
            );
            assert!(stderr.is_empty());
        }
    }

    #[test]
    fn interactive_keeps_trimming_stats_and_error_recovery() {
        for stop in ["ExIt", "qUiT"] {
            let input = format!("\n \t\n  first  \n fail \n\tsecond\t\n {stop} \nnever\n");
            let mut prompts = Vec::new();
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            run_chat_io(None, Cursor::new(input), &mut stdout, &mut stderr, |text| {
                prompts.push(text.to_string());
                if text == "fail" {
                    Err(InferenceError::InvalidInput("retry next line".into()))
                } else {
                    Ok(output(text))
                }
            })
            .unwrap();
            assert_eq!(prompts, ["first", "fail", "second"]);
            assert_eq!(
                stdout,
                b"first\n[4 prompt tokens, 2 generated]\nsecond\n[4 prompt tokens, 2 generated]\n"
            );
            let stderr = String::from_utf8(stderr).unwrap();
            assert!(stderr.starts_with("Model loaded. Type 'exit' or 'quit' to stop.\n\n"));
            assert_eq!(stderr.matches("Generation error:").count(), 1);
            assert!(stderr.contains("retry next line"));
        }
    }

    struct FailingInput;

    impl Read for FailingInput {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("input failed"))
        }
    }

    impl BufRead for FailingInput {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Err(io::Error::other("input failed"))
        }

        fn consume(&mut self, _amount: usize) {
            panic!("failed input has no bytes to consume");
        }
    }

    #[test]
    fn interactive_reports_input_error_and_stops() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        run_chat_io(None, FailingInput, &mut stdout, &mut stderr, |_| {
            panic!("failed input must not reach generation");
        })
        .unwrap();
        assert!(stdout.is_empty());
        assert!(
            String::from_utf8(stderr)
                .unwrap()
                .ends_with("Error reading input: input failed\n")
        );
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn oneshot_uses_checked_cpu_backend() {
        use super::Backend;
        use lattice_inference::GenerateConfig;
        use lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context;

        let mut backend = Backend::cpu(tiny_zero_model_with_context(32));
        let mut config = GenerateConfig::default();
        config.max_new_tokens = 0;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut calls = 0;
        let error = run_chat_io(
            Some(&"a".repeat(33)),
            NoInput,
            &mut stdout,
            &mut stderr,
            |text| {
                calls += 1;
                backend.generate_chat_line(text, &config)
            },
        )
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(error.contains("prompt (33 tokens) exceeds model context window (32)"));
        assert!(stdout.is_empty());
        assert!(stderr.is_empty());

        run_chat_io(
            Some(&"a".repeat(32)),
            NoInput,
            &mut stdout,
            &mut stderr,
            |text| {
                calls += 1;
                let output = backend.generate_chat_line(text, &config)?;
                assert_eq!(output.prompt_tokens, 32);
                assert_eq!(output.generated_tokens, 0);
                Ok(output)
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(stdout, b"\n");
        assert!(stderr.is_empty());
    }
}
