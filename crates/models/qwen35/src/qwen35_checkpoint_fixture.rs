//! Deterministic, synthetic Qwen3.5 checkpoint for serving startup tests.

use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Hidden width of the synthetic checkpoint.
pub const DIMENSIONS: usize = 32;

const INTERMEDIATE_SIZE: usize = 32;
const NUM_LAYERS: usize = 4;
const HEAD_DIM: usize = 256;
const LINEAR_KEY_HEAD_DIM: usize = 1;
const LINEAR_VALUE_HEAD_DIM: usize = 32;
const VOCAB_SIZE: usize = 7;
const TOKENIZER_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#;

const CONFIG_JSON: &str = r#"{
  "model_type": "qwen3_5",
  "hidden_size": 32,
  "num_hidden_layers": 4,
  "vocab_size": 7,
  "intermediate_size": 32,
  "rms_norm_eps": 0.000001,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 256,
  "rope_theta": 10000000.0,
  "partial_rotary_factor": 0.25,
  "linear_num_key_heads": 1,
  "linear_num_value_heads": 1,
  "linear_key_head_dim": 1,
  "linear_value_head_dim": 32,
  "linear_conv_kernel_dim": 4,
  "tie_word_embeddings": true,
  "mtp_num_hidden_layers": 0,
  "mtp_use_dedicated_embeddings": false,
  "full_attention_interval": 4,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
  "layer_mask": [true, true, true, true],
  "eos_token_id": 6,
  "max_position_embeddings": 1024
}"#;

struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    seed: u32,
    scale: f32,
    center: f32,
}

#[derive(Serialize)]
struct TensorHeader {
    dtype: &'static str,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn add_tensor(
    tensors: &mut Vec<TensorSpec>,
    name: String,
    shape: &[usize],
    seed: u32,
    scale: f32,
    center: f32,
) {
    tensors.push(TensorSpec {
        name,
        shape: shape.to_vec(),
        seed,
        scale,
        center,
    });
}

fn tensor_specs() -> Vec<TensorSpec> {
    let mut tensors = Vec::new();
    add_tensor(
        &mut tensors,
        "model.language_model.embed_tokens.weight".to_string(),
        &[VOCAB_SIZE, DIMENSIONS],
        1,
        1.0 / 4096.0,
        0.0,
    );
    add_tensor(
        &mut tensors,
        "model.language_model.norm.weight".to_string(),
        &[DIMENSIONS],
        2,
        1.0 / 8192.0,
        1.0,
    );

    for layer in 0..NUM_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        add_tensor(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            &[DIMENSIONS],
            10 + layer as u32,
            1.0 / 8192.0,
            1.0,
        );
        add_tensor(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            &[DIMENSIONS],
            20 + layer as u32,
            1.0 / 8192.0,
            1.0,
        );
        add_tensor(
            &mut tensors,
            format!("{prefix}.mlp.gate_proj.weight"),
            &[INTERMEDIATE_SIZE, DIMENSIONS],
            30 + layer as u32,
            1.0 / 4096.0,
            0.0,
        );
        add_tensor(
            &mut tensors,
            format!("{prefix}.mlp.up_proj.weight"),
            &[INTERMEDIATE_SIZE, DIMENSIONS],
            40 + layer as u32,
            1.0 / 4096.0,
            0.0,
        );
        add_tensor(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            &[DIMENSIONS, INTERMEDIATE_SIZE],
            50 + layer as u32,
            1.0 / 4096.0,
            0.0,
        );

        if layer < NUM_LAYERS - 1 {
            let linear = format!("{prefix}.linear_attn");
            let qkv_dim = 2 * LINEAR_KEY_HEAD_DIM + LINEAR_VALUE_HEAD_DIM;
            add_tensor(
                &mut tensors,
                format!("{linear}.in_proj_qkv.weight"),
                &[qkv_dim, DIMENSIONS],
                60 + layer as u32,
                1.0 / 4096.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.in_proj_z.weight"),
                &[LINEAR_VALUE_HEAD_DIM, DIMENSIONS],
                70 + layer as u32,
                1.0 / 4096.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.in_proj_b.weight"),
                &[1, DIMENSIONS],
                80 + layer as u32,
                1.0 / 8192.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.in_proj_a.weight"),
                &[1, DIMENSIONS],
                90 + layer as u32,
                1.0 / 8192.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.A_log"),
                &[1],
                100 + layer as u32,
                1.0 / 8192.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.dt_bias"),
                &[1],
                110 + layer as u32,
                1.0 / 8192.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.conv1d.weight"),
                &[qkv_dim, 1, 4],
                120 + layer as u32,
                1.0 / 4096.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.norm.weight"),
                &[LINEAR_VALUE_HEAD_DIM],
                130 + layer as u32,
                1.0 / 8192.0,
                1.0,
            );
            add_tensor(
                &mut tensors,
                format!("{linear}.out_proj.weight"),
                &[DIMENSIONS, LINEAR_VALUE_HEAD_DIM],
                140 + layer as u32,
                1.0 / 4096.0,
                0.0,
            );
        } else {
            let full = format!("{prefix}.self_attn");
            add_tensor(
                &mut tensors,
                format!("{full}.q_proj.weight"),
                &[2 * HEAD_DIM, DIMENSIONS],
                150,
                1.0 / 4096.0,
                0.0,
            );
            for projection in ["k_proj", "v_proj"] {
                add_tensor(
                    &mut tensors,
                    format!("{full}.{projection}.weight"),
                    &[HEAD_DIM, DIMENSIONS],
                    if projection == "k_proj" { 151 } else { 152 },
                    1.0 / 4096.0,
                    0.0,
                );
            }
            add_tensor(
                &mut tensors,
                format!("{full}.o_proj.weight"),
                &[DIMENSIONS, HEAD_DIM],
                153,
                1.0 / 4096.0,
                0.0,
            );
            add_tensor(
                &mut tensors,
                format!("{full}.q_norm.weight"),
                &[HEAD_DIM],
                154,
                1.0 / 8192.0,
                1.0,
            );
            add_tensor(
                &mut tensors,
                format!("{full}.k_norm.weight"),
                &[HEAD_DIM],
                155,
                1.0 / 8192.0,
                1.0,
            );
        }
    }
    tensors
}

fn checkpoint_bytes() -> Result<Vec<u8>, String> {
    let mut headers = BTreeMap::new();
    let mut data = Vec::new();
    for spec in tensor_specs() {
        let start = data.len();
        let mut state = spec.seed ^ 0xa5a5_5a5a;
        let element_count: usize = spec.shape.iter().product();
        for _ in 0..element_count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let centered = ((state >> 16) % 31) as i32 * 2 - 31;
            let value = spec.center + centered as f32 * spec.scale;
            data.extend_from_slice(&value.to_le_bytes());
        }
        headers.insert(
            spec.name,
            TensorHeader {
                dtype: "F32",
                shape: spec.shape,
                data_offsets: [start, data.len()],
            },
        );
    }
    let mut header = serde_json::to_vec(&headers)
        .map_err(|error| format!("serialize fixture tensor header: {error}"))?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let header_len =
        u64::try_from(header.len()).map_err(|error| format!("fixture header length: {error}"))?;
    let mut bytes = Vec::with_capacity(8 + header.len() + data.len());
    bytes.extend_from_slice(&header_len.to_le_bytes());
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

/// Write a deterministic Qwen3.5 checkpoint into an absent child directory.
pub fn write_checkpoint(dir: &Path) -> Result<(), String> {
    write_files(
        dir,
        &[
            ("config.json", CONFIG_JSON.as_bytes().to_vec()),
            ("model.safetensors", checkpoint_bytes()?),
            ("tokenizer.json", TOKENIZER_JSON.as_bytes().to_vec()),
        ],
    )
}
