//! Loader tests: checkpoints written in the published tensor layout with the synthetic
//! configuration of the decoder tests and random contents, and the tensor list compared with
//! the published checkpoint's header.

use std::path::{Path, PathBuf};

use lattice_arch::inkling2_config::{AttentionSpan, parse_config_json};
use lattice_inference_core::InferenceError;
use lattice_weights::f32_weights::SafetensorsFile;
use lattice_weights::half_bits::bf16_bits_to_f32;

use crate::tests::{CONFIG, config};
use crate::{
    ExpertBank, ExpertFormat, Inkling2Checkpoint, Inkling2Model, Inkling2Weights, KvConvolution,
    TensorSpec, expected_text_tensors,
};

const PROMPT: [u32; 5] = [3, 17, 5, 36, 9];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / 2_147_483_648.0 * 2.0 - 1.0
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 8) as u8
    }
}

/// One stored tensor: name, dtype label, shape and little-endian payload.
#[derive(Clone)]
pub(crate) struct Stored {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn formats(layers: usize) -> Vec<ExpertFormat> {
    (0..layers)
        .map(|layer| {
            if layer < 10 {
                ExpertFormat::Nvfp4
            } else {
                ExpertFormat::Mxfp8
            }
        })
        .collect()
}

fn bf16_bytes(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}

fn contents(spec: &TensorSpec, rng: &mut Rng) -> Vec<u8> {
    let count: usize = spec.shape.iter().product();
    let name = spec.name.as_str();
    let scale = name.ends_with("_scale");
    match spec.dtype {
        "BF16" => {
            let around_one = name.ends_with("norm.weight") || name.ends_with(".scale");
            (0..count)
                .flat_map(|_| {
                    let unit = rng.unit();
                    bf16_bytes(if around_one {
                        1.0 + 0.2 * unit
                    } else {
                        0.3 * unit
                    })
                })
                .collect()
        }
        "F32" => (0..count)
            .flat_map(|_| {
                let unit = rng.unit();
                let value = if name.ends_with("_amax") {
                    0.5 + 0.1 * unit
                } else if name.ends_with("_outer_scale_denominator") {
                    16.0
                } else {
                    unit
                };
                value.to_le_bytes()
            })
            .collect(),
        "U8" if scale => (0..count).map(|_| 116 + rng.byte() % 4).collect(),
        "U8" => (0..count).map(|_| rng.byte()).collect(),
        "F8_E4M3" if scale => (0..count).map(|_| 0x28 + rng.byte() % 16).collect(),
        "F8_E4M3" => (0..count)
            .map(|_| {
                let byte = rng.byte();
                if byte & 0x7f == 0x7f { byte ^ 1 } else { byte }
            })
            .collect(),
        other => panic!("no contents for dtype {other}"),
    }
}

pub(crate) fn checkpoint_tensors(seed: u64) -> Vec<Stored> {
    let c = config(3);
    let mut rng = Rng(seed);
    let mut stored: Vec<Stored> = expected_text_tensors(&c, &formats(c.num_hidden_layers))
        .expect("tensor list")
        .iter()
        .map(|spec| Stored {
            name: spec.name.clone(),
            dtype: spec.dtype.to_string(),
            shape: spec.shape.clone(),
            bytes: contents(spec, &mut rng),
        })
        .collect();
    stored.push(Stored {
        name: "model.visual.proj_in.weight".to_string(),
        dtype: "BF16".to_string(),
        shape: vec![2, 2],
        bytes: vec![0; 8],
    });
    stored
}

pub(crate) fn safetensors(tensors: &[Stored]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0;
    for (index, tensor) in tensors.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let dims: Vec<String> = tensor.shape.iter().map(usize::to_string).collect();
        let end = offset + tensor.bytes.len();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
            tensor.name,
            tensor.dtype,
            dims.join(",")
        ));
        offset = end;
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    for tensor in tensors {
        bytes.extend_from_slice(&tensor.bytes);
    }
    bytes
}

fn checkpoint(files: &[&[Stored]]) -> Result<Inkling2Checkpoint, InferenceError> {
    let files = files
        .iter()
        .map(|tensors| SafetensorsFile::from_bytes(safetensors(tensors)))
        .collect::<Result<Vec<_>, _>>()?;
    Inkling2Checkpoint::from_files(files)
}

fn load(tensors: &[Stored]) -> Result<Inkling2Weights, InferenceError> {
    Inkling2Weights::load(&checkpoint(&[tensors])?, &config(3))
}

fn stored<'a>(tensors: &'a [Stored], name: &str) -> &'a Stored {
    tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is stored"))
}

fn stored_mut<'a>(tensors: &'a mut [Stored], name: &str) -> &'a mut Stored {
    tensors
        .iter_mut()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is stored"))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]]))
        .collect()
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|p| bf16_bits_to_f32(u16::from_le_bytes([p[0], p[1]])))
        .collect()
}

fn row(values: &[f32], width: usize, index: usize) -> Vec<f32> {
    values[index * width..(index + 1) * width].to_vec()
}

pub(crate) fn scratch_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lattice_inkling2_{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

pub(crate) fn write(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(dir.join(name), bytes).expect("scratch file");
}

#[test]
fn the_tensor_list_matches_the_published_checkpoint_header() {
    let config = parse_config_json(include_str!("../tests/fixtures/inkling2_mini_config.json"))
        .expect("the published configuration parses")
        .text;
    let mut listed: Vec<(String, String, Vec<usize>)> =
        include_str!("../tests/fixtures/inkling2_mini_text_tensors.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                let fields: Vec<&str> = line.split(' ').collect();
                let [name, dtype, shape] = fields.as_slice() else {
                    panic!("malformed fixture line {line}");
                };
                let shape = shape
                    .split(',')
                    .map(|d| d.parse().expect("dimension"))
                    .collect();
                (name.to_string(), dtype.to_string(), shape)
            })
            .collect();
    assert_eq!(listed.len(), 822);
    let format_of = |layer: usize| {
        let name = format!("model.llm.layers.{layer}.mlp.experts.w13_weight");
        match listed.iter().find(|t| t.0 == name).map(|t| t.1.as_str()) {
            Some("U8") => ExpertFormat::Nvfp4,
            Some("F8_E4M3") => ExpertFormat::Mxfp8,
            other => panic!("{name}: {other:?}"),
        }
    };
    let formats: Vec<ExpertFormat> = (0..config.num_hidden_layers).map(format_of).collect();
    assert_eq!(
        formats
            .iter()
            .position(|&format| format == ExpertFormat::Mxfp8),
        Some(20)
    );
    let mut planned: Vec<(String, String, Vec<usize>)> = expected_text_tensors(&config, &formats)
        .expect("tensor list")
        .into_iter()
        .map(|spec| (spec.name, spec.dtype.to_string(), spec.shape))
        .collect();
    planned.sort();
    listed.sort();
    assert_eq!(planned, listed);
}

#[test]
fn a_checkpoint_in_the_published_layout_loads_and_runs() {
    let weights = load(&checkpoint_tensors(7)).expect("the checkpoint loads");
    let model = Inkling2Model::new(config(3), weights).expect("the weights fit the configuration");
    let logits = model
        .forward(&PROMPT, &mut model.new_cache())
        .expect("forward");
    assert_eq!(logits.len(), model.config().unpadded_vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn each_tensor_lands_where_the_forward_reads_it() {
    let c = config(3);
    let h = c.hidden_size;
    let lc = c.loop_config;
    let tensors = checkpoint_tensors(11);
    let weights = load(&tensors).expect("the checkpoint loads");
    let values = |name: &str| f32s(&stored(&tensors, name).bytes);
    let bf16_values = |name: &str| bf16s(&stored(&tensors, name).bytes);

    let queries = values("model.llm.middle_cycle_cross_pass_queries");
    let kv_queries = values("model.llm.middle_cycle_cross_pass_kv_queries");
    for j in 0..lc.core_end - lc.core_start {
        assert_eq!(weights.message_queries[4 * j], row(&queries, h, 2 * j));
        assert_eq!(
            weights.message_queries[4 * j + 1],
            row(&kv_queries, h, 2 * j)
        );
        assert_eq!(
            weights.message_queries[4 * j + 2],
            row(&kv_queries, h, 2 * j + 1)
        );
        assert_eq!(
            weights.message_queries[4 * j + 3],
            row(&queries, h, 2 * j + 1)
        );
    }

    assert!(weights.layers[0].attention_input_queries.is_empty());
    assert_eq!(c.layer_spans[1], AttentionSpan::Sliding);
    assert_eq!(
        weights.layers[1].attention_input_queries,
        vec![values("model.llm.attn_res.attn_queries.0")]
    );
    let global = (1..c.num_hidden_layers)
        .find(|&layer| c.layer_spans[layer] == AttentionSpan::Global)
        .expect("a global layer");
    let global_queries = values(&format!("model.llm.attn_res.attn_queries.{}", global - 1));
    assert_eq!(
        weights.layers[global].attention_input_queries,
        (0..3)
            .map(|i| row(&global_queries, h, i))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        weights.layers[global].mlp_input_query,
        values(&format!("model.llm.attn_res.mlp_queries.{global}"))
    );
    for (index, which) in ["q", "k", "v"].into_iter().enumerate() {
        assert_eq!(
            weights.layers[global].attention_norms[index],
            bf16_values(&format!(
                "model.llm.layers.{global}.{which}_input_norm.weight"
            ))
        );
    }
    match &weights.layers[global].attention.value_convolution {
        KvConvolution::Dynamic {
            static_taps,
            projection,
        } => {
            assert_eq!(
                *static_taps,
                bf16_values(&format!("model.llm.layers.{global}.attn.v_sconv.static_w"))
            );
            let proj = bf16_values(&format!(
                "model.llm.layers.{global}.attn.v_sconv.weight_proj"
            ));
            assert_eq!(projection.get(1, 2), Some(proj[h + 2]));
        }
        other => panic!("a global layer has a dynamic value convolution, not {other:?}"),
    }

    let layer = lc.core_start;
    let gates = weights.layers[layer]
        .gates
        .as_ref()
        .expect("a loop layer has message gates");
    let controller = |name: &str, part: &str| {
        bf16_values(&format!(
            "model.llm.layers.{layer}.controllers.{name}.{part}_weight"
        ))
    };
    assert_eq!(
        gates.query.down.get(0, 1),
        Some(controller("attn", "down")[1])
    );
    assert_eq!(
        gates.key.down.get(1, 0),
        Some(controller("k", "down")[2 * h])
    );
    assert_eq!(
        gates.value.forget.get(2, 1),
        Some(controller("v", "forget")[2 * lc.gate_rank + 1])
    );
    assert_eq!(gates.mlp.gain.get(0, 3), Some(controller("mlp", "gain")[3]));
    assert!(weights.layers[layer - 1].gates.is_none());

    let w2 = "model.llm.layers.0.mlp.experts.w2_weight";
    let expected = ExpertBank::nvfp4(
        c.n_routed_experts,
        h,
        c.moe_intermediate_size,
        stored(&tensors, w2).bytes.clone(),
        stored(&tensors, &format!("{w2}_scale")).bytes.clone(),
        values(&format!("{w2}_amax"))
            .iter()
            .map(|a| a / 16.0)
            .collect(),
    )
    .expect("expected bank");
    let x: Vec<f32> = (0..c.moe_intermediate_size)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    for expert in 0..c.n_routed_experts {
        let mut loaded = vec![0.0; h];
        let mut reference = vec![0.0; h];
        weights.layers[0]
            .moe
            .down
            .matvec(expert, &x, &mut loaded)
            .expect("loaded bank");
        expected
            .matvec(expert, &x, &mut reference)
            .expect("expected bank");
        assert_eq!(loaded, reference, "expert {expert}");
    }
}

type Damage = fn(&mut Vec<Stored>);

#[test]
fn a_damaged_checkpoint_is_refused_with_the_tensor_named() {
    let cases: [(&str, &str, Damage); 11] = [
        (
            "model.llm.layers.3.attn.wr_du.weight",
            "Missing tensor",
            |t| t.retain(|s| s.name != "model.llm.layers.3.attn.wr_du.weight"),
        ),
        (
            "model.llm.layers.2.mlp.gate.bias",
            "dtype BF16, expected F32",
            |t| {
                let s = stored_mut(t, "model.llm.layers.2.mlp.gate.bias");
                s.dtype = "BF16".to_string();
                s.bytes.truncate(s.bytes.len() / 2);
            },
        ),
        (
            "model.llm.embed_norm.weight",
            "shape [33], expected [32]",
            |t| {
                let s = stored_mut(t, "model.llm.embed_norm.weight");
                s.shape = vec![33];
                s.bytes.extend_from_slice(&[0x80, 0x3f]);
            },
        ),
        ("model.llm.layers.0.attn.q_bias", "does not read", |t| {
            t.push(Stored {
                name: "model.llm.layers.0.attn.q_bias".to_string(),
                dtype: "F32".to_string(),
                shape: vec![1],
                bytes: vec![0; 4],
            })
        }),
        (
            "model.llm.layers.0.mlp.experts.w2_weight_scale",
            "scale 5 has its sign bit set",
            |t| {
                stored_mut(t, "model.llm.layers.0.mlp.experts.w2_weight_scale").bytes[5] = 0xa8;
            },
        ),
        (
            "model.llm.layers.11.mlp.experts.w13_weight_scale",
            "scale 3 is 255",
            |t| {
                stored_mut(t, "model.llm.layers.11.mlp.experts.w13_weight_scale").bytes[3] = 0xff;
            },
        ),
        (
            "model.llm.layers.4.mlp.experts.w13_weight_outer_scale_denominator",
            "is not positive",
            |t| {
                stored_mut(
                    t,
                    "model.llm.layers.4.mlp.experts.w13_weight_outer_scale_denominator",
                )
                .bytes = 0.0f32.to_le_bytes().to_vec();
            },
        ),
        (
            "model.llm.layers.2.mlp.experts.w2_weight_amax",
            "expert 1 has amax",
            |t| {
                stored_mut(t, "model.llm.layers.2.mlp.experts.w2_weight_amax").bytes[4..8]
                    .copy_from_slice(&(-0.5f32).to_le_bytes());
            },
        ),
        (
            "model.llm.layers.1.attn.wq_du.weight",
            "element 2 is not finite",
            |t| {
                stored_mut(t, "model.llm.layers.1.attn.wq_du.weight").bytes[4..6]
                    .copy_from_slice(&0x7f80u16.to_le_bytes());
            },
        ),
        (
            "model.llm.attn_res.output_query",
            "element 0 is not finite",
            |t| {
                stored_mut(t, "model.llm.attn_res.output_query").bytes[0..4]
                    .copy_from_slice(&f32::NAN.to_le_bytes());
            },
        ),
        (
            "model.llm.layers.6.mlp.experts.w13_weight",
            "neither NVFP4",
            |t| {
                stored_mut(t, "model.llm.layers.6.mlp.experts.w13_weight").dtype = "I8".to_string();
            },
        ),
    ];
    let intact = checkpoint_tensors(5);
    load(&intact).expect("the undamaged checkpoint loads");
    for (name, problem, damage) in cases {
        let mut tensors = intact.clone();
        damage(&mut tensors);
        match load(&tensors) {
            Ok(_) => panic!("{name}: a damaged checkpoint loaded"),
            Err(error) => {
                let text = error.to_string();
                assert!(
                    text.contains(name) && text.contains(problem),
                    "{name}: expected {problem:?}, got {text}"
                );
            }
        }
    }
}

#[test]
fn a_checkpoint_split_over_files_loads_and_a_tensor_in_two_files_is_refused() {
    let tensors = checkpoint_tensors(13);
    let (first, second) = tensors.split_at(tensors.len() / 2);
    let split = checkpoint(&[first, second]).expect("two files");
    Inkling2Weights::load(&split, &config(3)).expect("a split checkpoint loads");

    let twice = [first.to_vec(), vec![first[0].clone()]];
    let refused = checkpoint(&[&twice[0], &twice[1]]).expect_err("a tensor stored twice");
    assert!(
        refused.to_string().contains(&first[0].name)
            && refused.to_string().contains("more than one file"),
        "{refused}"
    );
}

#[test]
fn a_checkpoint_directory_opens_with_and_without_an_index() {
    let tensors = checkpoint_tensors(17);
    let config_json = CONFIG.replace("PASSES", "3");

    let single = scratch_dir("single");
    write(&single, "config.json", config_json.as_bytes());
    write(&single, "model.safetensors", &safetensors(&tensors));
    let model = Inkling2Model::open(&single);
    let _ = std::fs::remove_dir_all(&single);
    let model = model.expect("a single-file checkpoint opens");
    assert!(
        model
            .forward(&PROMPT, &mut model.new_cache())
            .expect("forward")
            .iter()
            .all(|v| v.is_finite())
    );

    let sharded = scratch_dir("sharded");
    let (first, second) = tensors.split_at(tensors.len() / 2);
    let shard_names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    write(&sharded, "config.json", config_json.as_bytes());
    write(&sharded, shard_names[0], &safetensors(first));
    write(&sharded, shard_names[1], &safetensors(second));
    let entries: Vec<String> = first
        .iter()
        .map(|t| (t, shard_names[0]))
        .chain(second.iter().map(|t| (t, shard_names[1])))
        .map(|(t, shard)| format!("\"{}\":\"{shard}\"", t.name))
        .collect();
    let index = format!(
        "{{\"metadata\":{{}},\"weight_map\":{{{}}}}}",
        entries.join(",")
    );
    write(&sharded, "model.safetensors.index.json", index.as_bytes());
    let opened = Inkling2Model::open(&sharded).map(|_| ());

    let misplaced = index.replacen(
        &format!("\"{}\":\"{}\"", first[0].name, shard_names[0]),
        &format!("\"{}\":\"{}\"", first[0].name, shard_names[1]),
        1,
    );
    write(
        &sharded,
        "model.safetensors.index.json",
        misplaced.as_bytes(),
    );
    let refused = Inkling2Checkpoint::open(&sharded).map(|_| ());
    let _ = std::fs::remove_dir_all(&sharded);

    opened.expect("a sharded checkpoint opens");
    let refused = refused.expect_err("an index that misplaces a tensor");
    assert!(
        refused.to_string().contains(&first[0].name)
            && refused.to_string().contains("does not hold it"),
        "{refused}"
    );
}
