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
        gen_cfg: &lattice_inference::model::qwen35_config::GenerateConfig,
    ) -> Result<
        lattice_inference::model::qwen35_config::GenerateOutput,
        lattice_inference::error::InferenceError,
    > {
        self.state.generate(prompt, &self.tokenizer, gen_cfg)
    }
}

/// Narrow accessor for `serve.rs`: the Metal worker needs this cache-length
/// cap without reaching into `MetalChatBackend`'s internals across the
/// module boundary.
#[cfg(feature = "metal-gpu")]
pub(crate) fn chat_max_cache_len() -> usize {
    MetalChatBackend::MAX_CACHE_LEN
}

pub(crate) fn run_chat(
    model_path: &str,
    max_tokens: usize,
    temperature: f32,
    tokenizer_dir: Option<&str>,
) {
    use std::io::{BufRead, Write};
    use std::path::Path;

    let path = Path::new(model_path);
    let format = backend::detect_format(path);
    #[cfg(feature = "metal-gpu")]
    let tokenizer_dir_path = tokenizer_dir.map(Path::new);
    #[cfg(not(feature = "metal-gpu"))]
    let _ = tokenizer_dir;

    eprintln!("Loading model from {model_path}...");

    enum Backend {
        Cpu(Box<lattice_inference::model::qwen35::Qwen35Model>),
        #[cfg(feature = "metal-gpu")]
        Metal(Box<MetalChatBackend>),
    }

    let mut model = match format {
        backend::ModelFormat::Safetensors => {
            match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(path) {
                Ok(m) => Backend::Cpu(Box::new(m)),
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
    eprintln!("Model loaded. Type 'exit' or 'quit' to stop.\n");

    let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        temperature,
        ..Default::default()
    };

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let prompt = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading input: {e}");
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

        match &mut model {
            Backend::Cpu(m) => match m.generate(trimmed, &gen_cfg) {
                Ok(output) => {
                    let _ = writeln!(stdout, "{}", output.text);
                    let _ = writeln!(
                        stdout,
                        "[{} prompt tokens, {} generated]",
                        output.prompt_tokens, output.generated_tokens
                    );
                }
                Err(e) => {
                    eprintln!("Generation error: {e}");
                }
            },
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(m) => match m.generate(trimmed, &gen_cfg) {
                Ok(output) => {
                    let _ = writeln!(stdout, "{}", output.text);
                    let _ = writeln!(
                        stdout,
                        "[{} prompt tokens, {} generated]",
                        output.prompt_tokens, output.generated_tokens
                    );
                }
                Err(e) => {
                    eprintln!("Generation error: {e}");
                }
            },
        }
    }
}
