//! Dump EmbeddingGemma 2 multimodal stages for a corpus, for parity checks against a
//! reference implementation.
//!
//! Usage:
//!   cargo run -p lattice-inference --release --example dump_embeddinggemma2_vision -- \
//!     --model-dir <dir> --corpus <corpus.json> --out <out.safetensors> \
//!     [--widths 768,512,256,128] [--max-tokens 8192]
//!
//! `<corpus.json>` is `[{"id": "...", "parts": [{"text": "..."} | {"image":
//! {"raw_rgb8_path": "...", "width": N, "height": N}}]}, ...]`. Each `raw_rgb8_path`
//! holds exactly `width * height * 3` row-major RGB bytes; the example hashes each file as
//! it reads it and reports the digest in the row metadata so a caller can compare it
//! against the reference manifest.
//!
//! The output is a safetensors file of F32 tensors plus `__metadata__`. A row `<id>` with
//! `k` image parts writes the keys below, where `image_i` is the i-th image part in order.
//! Stage names match the reference capture of the same stage; where the reference keeps
//! padded rows, this dump stores the valid prefix and the comparison takes the reference's
//! valid prefix.
//!
//! | dump key | reference capture | shape |
//! | --- | --- | --- |
//! | `layout.input_ids` | `processor.input_ids` | `[sequence]` |
//! | `layout.valid_patch_counts` | `derived.valid_patch_counts` | `[k]` |
//! | `layout.soft_token_counts` | `derived.soft_token_counts` | `[k]` |
//! | `text.final` | `text.final` | `[sequence, 768]` |
//! | `readout.pre_normalization` | `readout.pre_normalization` | `[1, 768]` |
//! | `readout.normalized_<w>` | `readout.normalized_<w>` | `[1, w]` |
//! | `image_i/vision.patch` | `vision.patch` | `[valid, 768]` |
//! | `image_i/vision.layer_0` / `_8` / `_15` | the matching layer capture | `[valid, 768]` |
//! | `image_i/vision.encoder` | `vision.encoder` | `[valid, 768]` |
//! | `image_i/vision.pooler.values` | `vision.pooler.values` | `[soft, 768]` |
//! | `image_i/vision.pooler.occupancy` | `vision.pooler.occupancy` | `[soft]` (1.0/0.0) |
//! | `image_i/vision.valid` | `vision.valid` | `[soft, 768]` |
//! | `image_i/vision.projected` | `vision.projected` | `[soft, 512]` |
//! | `image_i/vision.layer_0_q_normed` / `_k_normed` | not captured | `[valid, heads, head]` |
//! | `image_i/vision.layer_0_q_rotated` / `_k_rotated` | not captured | `[valid, heads, head]` |
//!
//! The `layer_0_q_*`/`layer_0_k_*` pairs are extra: the reference captures neither, and
//! they feed the RoPE differential. `q_normed`/`k_normed` are pre-rotation.
//! `--max-tokens` bounds each text part including its own wrapping tokens (default 8192);
//! `--max-tokens 0` removes the bound.

use lattice_inference::model::embeddinggemma2::{
    EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS, EmbeddingGemma2Model, EmbeddingGemma2Part,
};
use lattice_inference::model::embeddinggemma2_image::EmbeddingGemma2Rgb8Image;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

struct Args {
    model_dir: PathBuf,
    corpus: PathBuf,
    out: PathBuf,
    widths: Vec<usize>,
    max_tokens: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut corpus = None;
    let mut out = None;
    let mut widths = vec![768, 512, 256, 128];
    let mut max_tokens = Some(EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
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
        max_tokens,
    })
}

/// Safetensors header and body for F32 tensors, written without an extra dependency.
#[derive(Default)]
struct TensorSink {
    header: BTreeMap<String, Value>,
    body: Vec<u8>,
    metadata: BTreeMap<String, Value>,
}

impl TensorSink {
    fn insert(&mut self, name: &str, values: &[f32], shape: &[usize]) {
        let start = self.body.len();
        for value in values {
            self.body.extend_from_slice(&value.to_le_bytes());
        }
        let end = self.body.len();
        self.header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}),
        );
    }

    fn insert_bool(&mut self, name: &str, values: &[bool], shape: &[usize]) {
        let as_f32: Vec<f32> = values.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        self.insert(name, &as_f32, shape);
    }

    fn write(self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let mut header = self.header;
        header.insert(
            "__metadata__".to_string(),
            Value::Object(self.metadata.into_iter().collect()),
        );
        let mut header_bytes = serde_json::to_vec(&header)?;
        while !header_bytes.len().is_multiple_of(8) {
            header_bytes.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header_bytes.len() + self.body.len());
        file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(&header_bytes);
        file.extend_from_slice(&self.body);
        std::fs::write(path, file)?;
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let model =
        EmbeddingGemma2Model::from_model_dir(&args.model_dir)?.with_max_tokens(args.max_tokens)?;
    let corpus: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&args.corpus)?)?;

    let mut sink = TensorSink::default();
    for item in &corpus {
        let id = item["id"]
            .as_str()
            .ok_or("corpus item without a string id")?;
        let parts_json = item["parts"]
            .as_array()
            .ok_or("corpus item without a parts array")?;
        let mut parts = Vec::with_capacity(parts_json.len());
        let mut source_digests = Vec::new();
        for part in parts_json {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                parts.push(EmbeddingGemma2Part::Text(text.to_string()));
                continue;
            }
            let image = part.get("image").ok_or("part without text or image")?;
            let path = image["raw_rgb8_path"]
                .as_str()
                .ok_or("image part without raw_rgb8_path")?;
            let width = image["width"].as_u64().ok_or("image part without width")? as usize;
            let height = image["height"]
                .as_u64()
                .ok_or("image part without height")? as usize;
            let bytes = std::fs::read(path)?;
            let expected = width * height * 3;
            if bytes.len() != expected {
                return Err(
                    format!("{path} holds {} bytes, expected {expected}", bytes.len()).into(),
                );
            }
            source_digests.push(format!(
                "{}:{}x{}:{}",
                path,
                width,
                height,
                hex(&Sha256::digest(&bytes))
            ));
            parts.push(EmbeddingGemma2Part::Image(EmbeddingGemma2Rgb8Image::new(
                width, height, bytes,
            )?));
        }

        let mut cancelled = || false;
        let encoding = model.encode_parts_at_widths(&parts, &args.widths, &mut cancelled)?;
        let embedding_dim = model.dimensions();
        let prefix = |name: &str| format!("{id}/{name}");

        let ids: Vec<f32> = encoding.input_ids.iter().map(|&id| id as f32).collect();
        let sequence = encoding.counts.sequence_length;
        sink.insert(&prefix("layout.input_ids"), &ids, &[sequence]);
        let valid_patches: Vec<f32> = encoding
            .image_stages
            .iter()
            .map(|stage| stage.valid_patches as f32)
            .collect();
        let soft_counts: Vec<f32> = encoding
            .image_stages
            .iter()
            .map(|stage| stage.soft_rows as f32)
            .collect();
        sink.insert(
            &prefix("layout.valid_patch_counts"),
            &valid_patches,
            &[valid_patches.len()],
        );
        sink.insert(
            &prefix("layout.soft_token_counts"),
            &soft_counts,
            &[soft_counts.len()],
        );
        sink.insert(
            &prefix("text.final"),
            &encoding.token_states,
            &[sequence, embedding_dim],
        );
        sink.insert(
            &prefix("readout.pre_normalization"),
            &encoding.pre_normalization,
            &[1, embedding_dim],
        );
        for (width, embedding) in args.widths.iter().zip(&encoding.embeddings) {
            sink.insert(
                &prefix(&format!("readout.normalized_{width}")),
                embedding,
                &[1, *width],
            );
        }

        for (image_index, stage) in encoding.image_stages.iter().enumerate() {
            let patches = stage.valid_patches;
            let soft = stage.soft_rows;
            let vision_hidden = stage.hidden_size;
            let heads = stage.heads;
            let head_dim = stage.head_dim;
            let at = |name: &str| prefix(&format!("image_{image_index}/{name}"));
            sink.insert(&at("vision.patch"), &stage.patch, &[patches, vision_hidden]);
            sink.insert(
                &at("vision.layer_0"),
                &stage.layer_0,
                &[patches, vision_hidden],
            );
            sink.insert(
                &at("vision.layer_8"),
                &stage.layer_8,
                &[patches, vision_hidden],
            );
            sink.insert(
                &at("vision.layer_15"),
                &stage.layer_15,
                &[patches, vision_hidden],
            );
            sink.insert(
                &at("vision.encoder"),
                &stage.encoder,
                &[patches, vision_hidden],
            );
            sink.insert(
                &at("vision.pooler.values"),
                &stage.pooler_values,
                &[soft, vision_hidden],
            );
            sink.insert_bool(
                &at("vision.pooler.occupancy"),
                &stage.pooler_occupancy,
                &[soft],
            );
            sink.insert(&at("vision.valid"), &stage.valid, &[soft, vision_hidden]);
            sink.insert(
                &at("vision.projected"),
                &stage.projected,
                &[soft, stage.text_hidden],
            );
            sink.insert(
                &at("vision.layer_0_q_normed"),
                &stage.layer_0_q_normed,
                &[patches, heads, head_dim],
            );
            sink.insert(
                &at("vision.layer_0_k_normed"),
                &stage.layer_0_k_normed,
                &[patches, heads, head_dim],
            );
            sink.insert(
                &at("vision.layer_0_q_rotated"),
                &stage.layer_0_q_rotated,
                &[patches, heads, head_dim],
            );
            sink.insert(
                &at("vision.layer_0_k_rotated"),
                &stage.layer_0_k_rotated,
                &[patches, heads, head_dim],
            );
            sink.metadata.insert(
                at("source_digest"),
                Value::String(source_digests.get(image_index).cloned().unwrap_or_default()),
            );
        }
        // safetensors metadata values must be strings: a numeric JSON value makes the
        // header unreadable to the `safetensors` reader the comparison uses.
        sink.metadata.insert(
            prefix("sequence_length"),
            Value::String(sequence.to_string()),
        );
        sink.metadata.insert(
            prefix("image_rows"),
            Value::String(encoding.counts.image_rows.to_string()),
        );
        eprintln!(
            "{id}: {sequence} tokens, {} image rows, {} images",
            encoding.counts.image_rows,
            encoding.image_stages.len()
        );
    }
    sink.write(&args.out)?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
