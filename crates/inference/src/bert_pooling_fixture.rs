//! Deterministic, synthetic BERT checkpoints for pooling integration controls.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Unequal-length inputs with vocabulary entries in the synthetic checkpoint.
pub const INPUTS: [&str; 3] = ["hello", "hello world", "quiet blue river"];
/// Hidden width shared by the two embedding model identities under test.
pub const DIMENSIONS: usize = 384;
/// Position-table and tokenizer limit, including special tokens.
pub const MAX_POSITIONS: usize = 16;

const VOCAB: &str = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nquiet\nblue\nriver\n";
const FILES: [&str; 3] = ["config.json", "model.safetensors", "vocab.txt"];

struct TensorSpec {
    name: &'static str,
    shape: &'static [usize],
    seed: u32,
    scale: f32,
    shift: f32,
    diagonal: f32,
}

const fn tensor(
    name: &'static str,
    shape: &'static [usize],
    seed: u32,
    scale: f32,
    shift: f32,
    diagonal: f32,
) -> TensorSpec {
    TensorSpec {
        name,
        shape,
        seed,
        scale,
        shift,
        diagonal,
    }
}

const TENSORS: [TensorSpec; 21] = [
    tensor(
        "embeddings.word_embeddings.weight",
        &[10, DIMENSIONS],
        1,
        1.0 / 64.0,
        0.0,
        0.0,
    ),
    tensor(
        "embeddings.position_embeddings.weight",
        &[MAX_POSITIONS, DIMENSIONS],
        2,
        1.0 / 4096.0,
        0.0,
        0.0,
    ),
    tensor(
        "embeddings.token_type_embeddings.weight",
        &[2, DIMENSIONS],
        3,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "embeddings.LayerNorm.weight",
        &[DIMENSIONS],
        4,
        1.0 / 2048.0,
        1.0,
        0.0,
    ),
    tensor(
        "embeddings.LayerNorm.bias",
        &[DIMENSIONS],
        5,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.attention.self.query.weight",
        &[DIMENSIONS, DIMENSIONS],
        6,
        1.0 / 262_144.0,
        0.0,
        0.125,
    ),
    tensor(
        "encoder.layer.0.attention.self.query.bias",
        &[DIMENSIONS],
        7,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.attention.self.key.weight",
        &[DIMENSIONS, DIMENSIONS],
        8,
        1.0 / 262_144.0,
        0.0,
        0.125,
    ),
    tensor(
        "encoder.layer.0.attention.self.key.bias",
        &[DIMENSIONS],
        9,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.attention.self.value.weight",
        &[DIMENSIONS, DIMENSIONS],
        10,
        1.0 / 262_144.0,
        0.0,
        0.75,
    ),
    tensor(
        "encoder.layer.0.attention.self.value.bias",
        &[DIMENSIONS],
        11,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.attention.output.dense.weight",
        &[DIMENSIONS, DIMENSIONS],
        12,
        1.0 / 262_144.0,
        0.0,
        0.75,
    ),
    tensor(
        "encoder.layer.0.attention.output.dense.bias",
        &[DIMENSIONS],
        13,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.attention.output.LayerNorm.weight",
        &[DIMENSIONS],
        14,
        1.0 / 2048.0,
        1.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.attention.output.LayerNorm.bias",
        &[DIMENSIONS],
        15,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.intermediate.dense.weight",
        &[16, DIMENSIONS],
        16,
        1.0 / 4096.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.intermediate.dense.bias",
        &[16],
        17,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.output.dense.weight",
        &[DIMENSIONS, 16],
        18,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.output.dense.bias",
        &[DIMENSIONS],
        19,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.output.LayerNorm.weight",
        &[DIMENSIONS],
        20,
        1.0 / 2048.0,
        1.0,
        0.0,
    ),
    tensor(
        "encoder.layer.0.output.LayerNorm.bias",
        &[DIMENSIONS],
        21,
        1.0 / 8192.0,
        0.0,
        0.0,
    ),
];

#[derive(Serialize)]
struct TensorHeader {
    dtype: &'static str,
    shape: &'static [usize],
    data_offsets: [usize; 2],
}

#[derive(Serialize)]
struct Config {
    model_type: &'static str,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    layer_norm_eps: f32,
}

fn config_bytes() -> Result<Vec<u8>, String> {
    serde_json::to_vec_pretty(&Config {
        model_type: "bert",
        vocab_size: 10,
        hidden_size: DIMENSIONS,
        num_hidden_layers: 1,
        num_attention_heads: 6,
        intermediate_size: 16,
        max_position_embeddings: MAX_POSITIONS,
        type_vocab_size: 2,
        layer_norm_eps: 1e-5,
    })
    .map_err(|error| format!("serialize fixture config: {error}"))
}

fn checkpoint_bytes() -> Result<Vec<u8>, String> {
    let mut header = BTreeMap::new();
    let mut data = Vec::new();
    for spec in &TENSORS {
        let start = data.len();
        let mut state = spec.seed ^ 0xa5a5_5a5a;
        for index in 0..spec.shape.iter().product() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let odd = ((state >> 16) % 31) as i32 * 2 - 31;
            let mut value = odd as f32 * spec.scale + spec.shift;
            if spec.shape.len() == 2 && index / spec.shape[1] == index % spec.shape[1] {
                value += spec.diagonal;
            }
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            spec.name,
            TensorHeader {
                dtype: "F32",
                shape: spec.shape,
                data_offsets: [start, data.len()],
            },
        );
    }
    let mut header = serde_json::to_vec(&header)
        .map_err(|error| format!("serialize fixture tensor header: {error}"))?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let length =
        u64::try_from(header.len()).map_err(|error| format!("fixture header length: {error}"))?;
    let mut bytes = Vec::with_capacity(8 + header.len() + data.len());
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    Ok(bytes)
}

fn write_files(dir: &Path, files: &[(&str, Vec<u8>)]) -> Result<(), String> {
    fs::create_dir(dir).map_err(|error| format!("reserve {}: {error}", dir.display()))?;
    let mut created: Vec<PathBuf> = Vec::new();
    let result = (|| {
        for (name, bytes) in files {
            let path = dir.join(name);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| format!("create {}: {error}", path.display()))?;
            created.push(path.clone());
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| format!("write {}: {error}", path.display()))?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        let mut cleanup = Vec::new();
        for path in created.iter().rev() {
            if let Err(cleanup_error) = fs::remove_file(path) {
                cleanup.push(format!("remove {}: {cleanup_error}", path.display()));
            }
        }
        if let Err(cleanup_error) = fs::remove_dir(dir) {
            cleanup.push(format!("remove {}: {cleanup_error}", dir.display()));
        }
        return if cleanup.is_empty() {
            Err(error)
        } else {
            Err(format!(
                "{error}; cleanup also failed: {}",
                cleanup.join("; ")
            ))
        };
    }
    Ok(())
}

/// Write a deterministic checkpoint into an absent child of an existing directory.
///
/// Destination reservation and file creation refuse replacement. On failure only
/// files created by this call are removed; this is not a crash-atomic publication.
pub fn write_checkpoint(dir: &Path) -> Result<(), String> {
    let files = [
        ("config.json", config_bytes()?),
        ("model.safetensors", checkpoint_bytes()?),
        ("vocab.txt", VOCAB.as_bytes().to_vec()),
    ];
    write_files(dir, &files)
}

/// SHA256 of the three required checkpoint files, keyed by their exact filenames.
pub fn checkpoint_hashes(dir: &Path) -> Result<BTreeMap<String, String>, String> {
    FILES
        .into_iter()
        .map(|name| {
            let path = dir.join(name);
            let bytes =
                fs::read(&path).map_err(|error| format!("hash {}: {error}", path.display()))?;
            Ok((name.to_string(), format!("{:x}", Sha256::digest(&bytes))))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BertModel;
    use crate::pool::BertPooling;
    use crate::weights::SafetensorsFile;

    #[test]
    fn checkpoint_is_deterministic_and_loads_all_twenty_one_tensors() {
        let parent = tempfile::tempdir().unwrap();
        let first = parent.path().join("first");
        let second = parent.path().join("second");
        write_checkpoint(&first).unwrap();
        write_checkpoint(&second).unwrap();
        let hashes = checkpoint_hashes(&first).unwrap();
        assert_eq!(hashes, checkpoint_hashes(&second).unwrap());
        assert_eq!(hashes.len(), 3);
        assert!(hashes.values().all(|hash| hash.len() == 64));
        assert_eq!(
            fs::metadata(first.join("model.safetensors")).unwrap().len(),
            2_470_632
        );
        let weights = SafetensorsFile::open(&first.join("model.safetensors")).unwrap();
        assert_eq!(weights.tensor_names().len(), TENSORS.len());
        for spec in &TENSORS {
            assert_eq!(weights.tensor_dtype(spec.name), Some("F32"));
            let (values, shape) = weights.get_f32_tensor(spec.name).unwrap();
            assert_eq!(shape, spec.shape);
            assert!(values.iter().all(|v| v.is_finite() && *v != 0.0));
        }
        let (tokens, _) = weights
            .get_f32_tensor("embeddings.word_embeddings.weight")
            .unwrap();
        for left in 0..10 {
            for right in left + 1..10 {
                let a = &tokens[left * DIMENSIONS..(left + 1) * DIMENSIONS];
                let b = &tokens[right * DIMENSIONS..(right + 1) * DIMENSIONS];
                assert!((1..DIMENSIONS).any(|column| a[0] * b[column] != b[0] * a[column]));
            }
        }
        let model = BertModel::from_directory(&first).unwrap();
        assert_eq!(model.dimensions(), DIMENSIONS);
        assert_eq!(model.config().num_hidden_layers, 1);
        assert_eq!(model.config().num_attention_heads, 6);
        assert_eq!(model.config().intermediate_size, 16);
        assert_eq!(model.config().max_position_embeddings, MAX_POSITIONS);
        for (input, ids) in
            INPUTS
                .into_iter()
                .zip([vec![2, 5, 3], vec![2, 5, 6, 3], vec![2, 7, 8, 9, 3]])
        {
            let actual = model.tokenizer().tokenize(input);
            assert_eq!(&actual.input_ids[..actual.real_length], ids);
            assert_eq!(actual.pre_truncation_len, ids.len());
        }
    }

    #[test]
    fn checkpoint_makes_pooling_and_input_order_observable() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("checkpoint");
        write_checkpoint(&dir).unwrap();
        let mut model = BertModel::from_directory(&dir).unwrap();
        let mut by_mode = Vec::new();
        for mode in [BertPooling::CLS, BertPooling::Mean] {
            model.set_pooling(mode);
            let vectors: Vec<_> = INPUTS
                .into_iter()
                .map(|text| model.encode(text).unwrap())
                .collect();
            for vector in &vectors {
                assert_eq!(vector.len(), DIMENSIONS);
                assert!(vector.iter().all(|v| v.is_finite()));
                let norm: f32 = vector.iter().map(|v| v * v).sum();
                assert!((norm - 1.0).abs() < 1e-5);
            }
            for left in 0..vectors.len() {
                for right in left + 1..vectors.len() {
                    assert!(
                        vectors[left]
                            .iter()
                            .zip(&vectors[right])
                            .any(|(a, b)| (a - b).abs() >= 1e-3)
                    );
                }
            }
            by_mode.push(vectors);
        }
        for input in 0..INPUTS.len() {
            let distance: f32 = by_mode[0][input]
                .iter()
                .zip(&by_mode[1][input])
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            assert!(distance.sqrt() >= 1e-3);
        }
    }

    #[test]
    fn checkpoint_refuses_existing_destination_without_replacement() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("checkpoint");
        write_checkpoint(&dir).unwrap();
        let before = checkpoint_hashes(&dir).unwrap();
        assert!(write_checkpoint(&dir).unwrap_err().contains("reserve"));
        assert_eq!(checkpoint_hashes(&dir).unwrap(), before);
        let file = parent.path().join("keep");
        fs::write(&file, b"unchanged").unwrap();
        assert!(write_checkpoint(&file).is_err());
        assert_eq!(fs::read(file).unwrap(), b"unchanged");
        assert!(checkpoint_hashes(&parent.path().join("absent")).is_err());
    }

    #[test]
    fn checkpoint_failed_write_removes_only_owned_files() {
        let parent = tempfile::tempdir().unwrap();
        let keep = parent.path().join("keep");
        fs::write(&keep, b"parent-owned").unwrap();
        let dir = parent.path().join("checkpoint");
        let error = write_files(
            &dir,
            &[
                ("created", b"first".to_vec()),
                ("missing/child", b"second".to_vec()),
            ],
        )
        .unwrap_err();
        assert!(error.contains("create") && error.contains("missing/child"));
        assert!(!dir.exists());
        assert_eq!(fs::read(keep).unwrap(), b"parent-owned");
    }
}
