//! `lattice` CLI - interactive chat, HTTP serve, and preflight subcommands. See [docs/capability-matrix.md](../../../../../docs/capability-matrix.md).
//!
//! # Usage
//!
//! ```text
//! lattice chat --model /path/to/model [--max-tokens 256] [--temperature 0.7]
//! lattice serve --model /path/to/model [--host 127.0.0.1] [--port 8080] [--max-tokens 256]
//! lattice doctor --model /path/to/model [--context 4096]
//! lattice prune-score --q4-dir /path/to/model-q4 --tokenizer-dir /path/to/model \
//!   --calibration-corpus calibration.txt --validation-corpus validation.txt \
//!   --prune-layers 4 --output lattice_pruning.json
//! ```

use clap::{Parser, Subcommand};

mod chat;
mod doctor;
mod prune_score;
mod serve;

#[derive(Parser)]
#[command(name = "lattice", about = "Pure-Rust transformer inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive chat with a model
    Chat {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Maximum tokens to generate per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
    /// Start HTTP server with OpenAI-compatible API
    Serve {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Host address to bind (default: 127.0.0.1; use 0.0.0.0 for LAN)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Maximum tokens to generate per request (default when request omits max_tokens)
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Model identifier echoed in responses (defaults to the model path basename)
        #[arg(long)]
        model_id: Option<String>,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
        /// Cap on outstanding (queued + in-flight) requests to the Metal GPU
        /// worker (issue #932) before new requests are rejected with HTTP
        /// 503. Only applies to Q4/Metal-backed serving; the CPU backend has
        /// no shared worker queue to bound. Conservative default: this
        /// worker serializes all generation onto one dedicated thread, so a
        /// deep queue just means memory growth with no throughput benefit.
        /// Must be between 1 and `tokio::sync::Semaphore::MAX_PERMITS`
        /// (issue #939): zero would admit nothing (every request fails
        /// admission), and clap rejects anything larger here instead of
        /// deferring to `MetalWorker::spawn`'s own
        /// `Semaphore::new`-precondition panic.
        #[arg(
            long,
            default_value = "32",
            value_parser = clap::builder::RangedU64ValueParser::<usize>::new()
                .range(1..=(tokio::sync::Semaphore::MAX_PERMITS as u64))
        )]
        max_pending: usize,
    },
    /// Preflight check: memory fit and artifact compatibility, without
    /// loading any model weights (config + tensor index inspection only).
    Doctor {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Context length to check feasibility for. When omitted, only the
        /// maximum feasible context length is reported.
        #[arg(long)]
        context: Option<usize>,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
    /// Score layer importance on a calibration corpus and PPL-gate a pruning plan.
    PruneScore {
        #[command(flatten)]
        args: prune_score::Args,
    },
}

// ---------------------------------------------------------------------------
// backend: model-directory format detection + Q4/Metal loading
//
// `lattice chat`/`lattice serve` originally only understood a safetensors
// directory (`model.safetensors` or a sharded index). Native Q4 quantized
// directories (per-tensor `.q4` files, the output of `quantize_q4`) route to
// the Metal GPU forward pass instead. Safetensors directories are completely
// unaffected: `detect_format` returns `Safetensors` for them exactly as
// before, and the safetensors load path is untouched.
//
// The detector itself (`ModelFormat` + `detect_format` + the two error
// message helpers) now lives in `lattice_inference::model_format` (ADR-080
// amendment, #829): it is shared, unmodified, with `lattice_serve.rs` and
// `chat_metal.rs`, which cannot see a `pub(crate)` item defined in this
// binary's own crate root. `backend` remains a crate-root alias so every
// sibling command module resolves the same shared detector.
// ---------------------------------------------------------------------------

use lattice_inference::model_format as backend;

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Chat {
            model,
            max_tokens,
            temperature,
            tokenizer_dir,
        } => {
            chat::run_chat(&model, max_tokens, temperature, tokenizer_dir.as_deref());
        }
        Command::Serve {
            model,
            host,
            port,
            max_tokens,
            model_id,
            tokenizer_dir,
            max_pending,
        } => {
            serve::run(
                model,
                host,
                port,
                max_tokens,
                model_id,
                tokenizer_dir,
                max_pending,
            )
            .await;
        }
        Command::Doctor {
            model,
            context,
            tokenizer_dir,
        } => {
            doctor::run(model, context, tokenizer_dir);
        }
        Command::PruneScore { args } => match prune_score::run(&args) {
            Ok(true) => {}
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    }
}

// ─── #939 CLI boundary tests: `--max-pending` range validation ────────────
//
// clap's own `value_parser!(usize).range(1..=Semaphore::MAX_PERMITS)` on the
// `Serve::max_pending` field (above) is the ONLY validation this binary
// needs for zero / too-large values -- unlike `lattice_serve.rs`'s hand
// rolled argv parser, clap already rejects a malformed string (`abc`,
// `-1`) itself, before this range check ever runs. These tests exercise
// that `value_parser` wiring directly through `Cli::try_parse_from`,
// rather than duplicating the range logic anywhere in this binary.
#[cfg(test)]
mod max_pending_cli_tests {
    use super::*;

    fn parse_max_pending(args: &[&str]) -> Result<usize, clap::Error> {
        let mut full = vec!["lattice", "serve", "--model", "/tmp/model"];
        full.extend_from_slice(args);
        match Cli::try_parse_from(full)?.command {
            Command::Serve { max_pending, .. } => Ok(max_pending),
            _ => panic!("expected Command::Serve, got a different Command variant"),
        }
    }

    #[test]
    fn max_pending_omitted_defaults_to_32() {
        assert_eq!(parse_max_pending(&[]).expect("no --max-pending"), 32);
    }

    #[test]
    fn max_pending_zero_is_rejected() {
        parse_max_pending(&["--max-pending", "0"])
            .expect_err("0 admits nothing and must be rejected, not silently accepted");
    }

    #[test]
    fn max_pending_one_above_max_permits_is_rejected() {
        let too_big = (tokio::sync::Semaphore::MAX_PERMITS as u128 + 1).to_string();
        parse_max_pending(&["--max-pending", &too_big]).expect_err(
            "Semaphore::MAX_PERMITS + 1 must be rejected before it can panic Semaphore::new",
        );
    }

    #[test]
    fn max_pending_at_max_permits_is_accepted() {
        let at_max = tokio::sync::Semaphore::MAX_PERMITS.to_string();
        assert_eq!(
            parse_max_pending(&["--max-pending", &at_max])
                .expect("Semaphore::MAX_PERMITS itself is the inclusive upper bound"),
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }

    #[test]
    fn max_pending_negative_is_rejected() {
        parse_max_pending(&["--max-pending", "-1"])
            .expect_err("a negative value must be rejected, not silently defaulted");
    }

    #[test]
    fn max_pending_malformed_is_rejected() {
        parse_max_pending(&["--max-pending", "not-a-number"])
            .expect_err("a non-numeric value must be rejected, not silently defaulted");
    }

    #[test]
    fn max_pending_valid_override_is_accepted() {
        assert_eq!(
            parse_max_pending(&["--max-pending", "8"]).expect("8 is a valid cap"),
            8
        );
    }
}
