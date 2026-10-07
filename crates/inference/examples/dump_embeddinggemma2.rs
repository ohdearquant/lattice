//! Dump EmbeddingGemma 2 token ids and embeddings for a corpus, for parity checks against a
//! reference implementation.
//!
//! Usage:
//!   cargo run -p lattice-inference --release --example dump_embeddinggemma2 -- \
//!     --model-dir <dir> --corpus <corpus.json> --out <out.json> \
//!     [--widths 768,512,256,128] [--hidden-ids id1,id2] [--max-tokens 8192] [--metal]
//!
//! `<corpus.json>` is `[{"id": "...", "text": "..."}, ...]`. The output is
//! `{"rows": [{"id", "ids", "emb": {"<width>": [...]}}]}`. Rows named by `--hidden-ids` also
//! carry `"hidden"`, the per-token states after the final projection (`[tokens][dim]`).
//! Each text is embedded exactly as given, so a task prefix must already be part of `text`.
//! `--max-tokens` bounds the token sequence including its beginning and end tokens (default 8192);
//! `--max-tokens 0` removes the bound. `--metal` runs the forward pass on the GPU in f32 (macOS with
//! the `metal-gpu` feature); it holds the machine GPU lock for the whole run.

use lattice_inference::model::embeddinggemma2::{
    EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS, EmbeddingGemma2Model,
};
use serde_json::{Value, json};
use std::path::PathBuf;

struct Args {
    model_dir: PathBuf,
    corpus: PathBuf,
    out: PathBuf,
    widths: Vec<usize>,
    hidden_ids: Vec<String>,
    max_tokens: Option<usize>,
    metal: bool,
}

type Forward<'a> = dyn FnMut(
        &[u32],
        &[usize],
        bool,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<f32>>), Box<dyn std::error::Error>>
    + 'a;

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut corpus = None;
    let mut out = None;
    let mut widths = vec![768, 512, 256, 128];
    let mut hidden_ids = Vec::new();
    let mut max_tokens = Some(EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS);
    let mut metal = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--metal" {
            metal = true;
            continue;
        }
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--corpus" => corpus = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--widths" => {
                widths = value
                    .split(',')
                    .map(|w| {
                        w.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("bad width {w:?}: {e}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--hidden-ids" => hidden_ids = value.split(',').map(str::to_string).collect(),
            "--max-tokens" => {
                let n = value
                    .parse::<usize>()
                    .map_err(|e| format!("bad --max-tokens {value:?}: {e}"))?;
                max_tokens = (n != 0).then_some(n);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        model_dir: model_dir.ok_or("--model-dir is required")?,
        corpus: corpus.ok_or("--corpus is required")?,
        out: out.ok_or("--out is required")?,
        widths,
        hidden_ids,
        max_tokens,
        metal,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let model =
        EmbeddingGemma2Model::from_model_dir(&args.model_dir)?.with_max_tokens(args.max_tokens)?;
    if args.metal {
        return dump_metal(&args, &model);
    }
    dump(&args, &model, &mut |ids, widths, want_hidden| {
        let embeddings = model.encode_ids_at_widths(ids, widths)?;
        let hidden = want_hidden.then(|| model.token_states(ids)).transpose()?;
        Ok((embeddings, hidden))
    })
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn dump_metal(args: &Args, model: &EmbeddingGemma2Model) -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_embeddinggemma2::MetalEmbeddingGemma2State;
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let mut state = MetalEmbeddingGemma2State::new(model)?;
    dump(args, model, &mut |ids, widths, want_hidden| {
        let embeddings = model.encode_ids_at_widths_metal(&mut state, ids, widths)?;
        let hidden = want_hidden
            .then(|| model.token_states_metal(&mut state, ids))
            .transpose()?;
        Ok((embeddings, hidden))
    })
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn dump_metal(
    _args: &Args,
    _model: &EmbeddingGemma2Model,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("--metal requires macOS and the metal-gpu feature".into())
}

fn dump(
    args: &Args,
    model: &EmbeddingGemma2Model,
    forward: &mut Forward<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let corpus: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&args.corpus)?)?;

    let mut rows = Vec::with_capacity(corpus.len());
    for item in &corpus {
        let id = item["id"]
            .as_str()
            .ok_or("corpus item without a string id")?;
        let text = item["text"]
            .as_str()
            .ok_or("corpus item without a string text")?;
        let ids = model.tokenize(text)?;
        let want_hidden = args.hidden_ids.iter().any(|h| h == id);
        let (embeddings, hidden) = forward(&ids, &args.widths, want_hidden)?;
        let emb: serde_json::Map<String, Value> = args
            .widths
            .iter()
            .zip(&embeddings)
            .map(|(width, v)| (width.to_string(), json!(v)))
            .collect();
        let mut row = json!({ "id": id, "ids": ids, "emb": emb });
        if let Some(states) = hidden {
            let dim = model.dimensions();
            let hidden: Vec<&[f32]> = states.chunks_exact(dim).collect();
            row["hidden"] = json!(hidden);
        }
        eprintln!("{id}: {} tokens", ids.len());
        rows.push(row);
    }
    std::fs::write(&args.out, serde_json::to_string(&json!({ "rows": rows }))?)?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
