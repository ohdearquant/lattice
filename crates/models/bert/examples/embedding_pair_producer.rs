//! Embed JSON batches with a local BERT model for comparison with reference records.
use lattice_compute::pool::BertPooling;
use lattice_model_bert::BertModel;
use lattice_tokenizer::common::load_tokenizer;
use serde_json::{Value, json};
use std::io::{self, BufRead, Read, Write};
use std::path::Path;

// JSON bytes per request, excluding the newline; mirrored by the Python client.
const MAX_REQUEST_BYTES: usize = 2_200_000;

fn read_request(input: &mut impl BufRead) -> io::Result<Option<String>> {
    let mut bytes = Vec::new();
    let count = input
        .take((MAX_REQUEST_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
    }
    if bytes.len() > MAX_REQUEST_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request too large",
        ));
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

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
    let mut input = stdin.lock();
    while let Some(line) = read_request(&mut input)? {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn refuses_oversize_unterminated_request_without_consuming_the_rest() {
        for extra in [1, 10] {
            let mut source = Cursor::new(vec![b'x'; MAX_REQUEST_BYTES + extra]);
            let error = read_request(&mut source).unwrap_err();
            assert_eq!(error.to_string(), "request too large");
            assert_eq!(source.position(), (MAX_REQUEST_BYTES + 1) as u64);
        }
    }

    #[test]
    fn accepts_at_limit_and_reads_multiple_lines() {
        let mut bytes = vec![b'x'; MAX_REQUEST_BYTES];
        bytes.extend_from_slice(b"\nnext\n");
        let mut source = Cursor::new(bytes);
        assert_eq!(
            read_request(&mut source).unwrap().unwrap().len(),
            MAX_REQUEST_BYTES
        );
        assert_eq!(read_request(&mut source).unwrap().as_deref(), Some("next"));
        assert!(read_request(&mut source).unwrap().is_none());
    }
}
