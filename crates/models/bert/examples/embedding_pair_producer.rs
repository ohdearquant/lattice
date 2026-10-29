//! Local-only BERT embedding producer for the offline #84 correctness pairing.
use lattice_compute::pool::BertPooling;
use lattice_model_bert::BertModel;
use lattice_tokenizer::common::load_tokenizer;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::path::Path;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let dir = args.next().ok_or("expected one model directory")?;
    if args.next().is_some() {
        return Err("expected one model directory".into());
    }
    let dir = Path::new(&dir);
    let tokenizer = load_tokenizer(dir)?;
    let mut model = BertModel::from_directory(dir)?;
    if model.dimensions() != 384
        || model.config().max_position_embeddings != 512
        || tokenizer.max_seq_len() != 512
    {
        return Err("unexpected BGE dimension or token limit".into());
    }
    model.set_pooling(BertPooling::CLS);
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.len() > 2_200_000 {
            return Err("request too large".into());
        }
        let request: Vec<String> = serde_json::from_str(&line)?;
        if request.is_empty()
            || request.len() > 32
            || request.iter().any(|x| x.is_empty() || x.len() > 65_536)
        {
            return Err("invalid batch".into());
        }
        let texts: Vec<&str> = request.iter().map(String::as_str).collect();
        let tokenized = if texts.len() == 1 {
            vec![tokenizer.tokenize(texts[0])]
        } else {
            tokenizer.tokenize_batch(&texts)
        };
        if tokenized.len() != texts.len() {
            return Err("token count mismatch".into());
        }
        let mut ids = Vec::with_capacity(texts.len());
        for item in tokenized {
            let n = item.real_length;
            if n == 0
                || n > 512
                || n != item.pre_truncation_len
                || item.input_ids.len() < n
                || item.attention_mask.len() < n
                || item.token_type_ids.len() < n
                || item.attention_mask[..n].iter().any(|x| *x != 1)
                || item.token_type_ids[..n].iter().any(|x| *x != 0)
            {
                return Err("truncated or invalid tokenization".into());
            }
            ids.push(item.input_ids[..n].to_vec());
        }
        let vectors = model.encode_batch(&texts)?;
        if vectors.len() != texts.len()
            || vectors
                .iter()
                .any(|v| v.len() != 384 || v.iter().any(|x| !x.is_finite()))
        {
            return Err("invalid embedding shape or value".into());
        }
        let response: Value = json!({"ids": ids, "vectors": vectors});
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("embedding producer refused: {error}");
        std::process::exit(2);
    }
}
