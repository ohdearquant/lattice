//! Reads the text decoder's weights from a checkpoint in the published Inkling 2 layout
//! (ADR-106, "Expert layout in the checkpoint" and "Checkpoint tensor names").
//!
//! Every tensor the decoder reads is listed first, from the configuration and each layer's
//! expert format, and the checkpoint is checked against that list (names, dtypes, shapes, and no
//! unread text-decoder tensor) before anything is copied out of it.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use lattice_arch::inkling2_config::{AttentionSpan, Inkling2TextConfig, parse_config};
use lattice_inference_core::InferenceError;
use lattice_weights::__internal::safetensors_from_open_file;
use lattice_weights::f32_weights::{SafetensorsFile, open_manifest_entry_once, parse_index};
use lattice_weights::half_bits::bf16_bits_to_f32;
use lattice_weights::mxfp8::MXFP8_BLOCK_LEN;
use lattice_weights::nvfp4::NVFP4_BLOCK_LEN;

use crate::model::Inkling2Model;
use crate::tensor::{DenseMatrix, ExpertBank};
use crate::weights::{
    AttentionWeights, Inkling2Weights, KvConvolution, LayerWeights, LoopGates, MessageGateWeights,
    MoeWeights,
};

/// Prefix of every text-decoder tensor in the checkpoint.
const TEXT_PREFIX: &str = "model.llm.";
const BF16: &str = "BF16";
const F32: &str = "F32";
const U8: &str = "U8";
const F8_E4M3: &str = "F8_E4M3";

/// Storage format of one layer's routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFormat {
    /// E2M1 codes stored as `U8` pairs, a UE4M3 scale per 16 elements of a row and a
    /// per-expert multiplier `amax / outer_scale_denominator`.
    Nvfp4,
    /// E4M3 values with an E8M0 scale per 32 elements of a row.
    Mxfp8,
}

/// A tensor the text decoder reads: its checkpoint name, safetensors dtype label and shape.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorSpec {
    /// Full checkpoint name, for example `model.llm.layers.0.attn.wq_du.weight`.
    pub name: String,
    /// Safetensors dtype label, for example `BF16`.
    pub dtype: &'static str,
    /// Shape as the checkpoint stores it.
    pub shape: Vec<usize>,
}

fn spec(name: String, dtype: &'static str, shape: &[usize]) -> TensorSpec {
    TensorSpec {
        name,
        dtype,
        shape: shape.to_vec(),
    }
}

fn top(suffix: &str) -> String {
    format!("{TEXT_PREFIX}{suffix}")
}

fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("{TEXT_PREFIX}layers.{layer}.{suffix}")
}

fn invalid(message: String) -> InferenceError {
    InferenceError::InvalidInput(message)
}

fn bad_tensor(name: &str, problem: &str) -> InferenceError {
    InferenceError::InvalidSafetensors(format!("tensor {name}: {problem}"))
}

fn in_tensor(name: &str, error: InferenceError) -> InferenceError {
    InferenceError::InvalidSafetensors(format!("tensor {name}: {error}"))
}

fn blocks(width: usize, block: usize, what: &str, layer: usize) -> Result<usize, InferenceError> {
    if width.is_multiple_of(block) {
        Ok(width / block)
    } else {
        Err(invalid(format!(
            "layer {layer}: the {what} {width} is not a multiple of the {block}-element scale block"
        )))
    }
}

fn expert_specs(
    config: &Inkling2TextConfig,
    layer: usize,
    format: ExpertFormat,
) -> Result<Vec<TensorSpec>, InferenceError> {
    let h = config.hidden_size;
    let experts = config.n_routed_experts;
    let intermediate = config.moe_intermediate_size;
    let mut specs = Vec::new();
    for (which, rows, cols, what) in [
        ("w13", 2 * intermediate, h, "hidden size"),
        ("w2", h, intermediate, "expert intermediate size"),
    ] {
        let weight = layer_tensor(layer, &format!("mlp.experts.{which}_weight"));
        match format {
            ExpertFormat::Nvfp4 => {
                let scale_cols = blocks(cols, NVFP4_BLOCK_LEN, what, layer)?;
                specs.push(spec(weight.clone(), U8, &[experts, rows, cols / 2]));
                specs.push(spec(
                    format!("{weight}_scale"),
                    F8_E4M3,
                    &[experts, rows, scale_cols],
                ));
                specs.push(spec(format!("{weight}_amax"), F32, &[experts]));
                specs.push(spec(format!("{weight}_outer_scale_denominator"), F32, &[1]));
            }
            ExpertFormat::Mxfp8 => {
                let scale_cols = blocks(cols, MXFP8_BLOCK_LEN, what, layer)?;
                specs.push(spec(weight.clone(), F8_E4M3, &[experts, rows, cols]));
                specs.push(spec(
                    format!("{weight}_scale"),
                    U8,
                    &[experts, rows, scale_cols],
                ));
            }
        }
    }
    Ok(specs)
}

/// The tensors the text decoder reads from a checkpoint for `config`, given each stored layer's
/// expert format. Tensors outside the text decoder (the vision tower) are not listed.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] when `formats` does not hold one entry per stored
/// layer, or when a width a packed format divides into scale blocks is not a multiple of the
/// block length.
pub fn expected_text_tensors(
    config: &Inkling2TextConfig,
    formats: &[ExpertFormat],
) -> Result<Vec<TensorSpec>, InferenceError> {
    let layers = config.num_hidden_layers;
    if formats.len() != layers {
        return Err(invalid(format!(
            "{} expert formats were given for {layers} layers",
            formats.len()
        )));
    }
    let h = config.hidden_size;
    let kernel = config.conv_kernel_size;
    let lc = config.loop_config;
    let mut specs = Vec::new();
    for (layer, &format) in formats.iter().enumerate() {
        let span = config
            .span(layer)
            .ok_or_else(|| invalid(format!("layer {layer} has no attention span")))?;
        let heads = config.heads(span);
        let q = heads.num_attention_heads * heads.head_dim;
        let kv = heads.num_key_value_heads * heads.head_dim;
        let r = heads.num_attention_heads * config.d_rel;
        let n = |suffix: &str| layer_tensor(layer, suffix);

        specs.push(spec(n("attn.wq_du.weight"), BF16, &[q, h]));
        specs.push(spec(
            n("attn.q_l2_clip.scale"),
            BF16,
            &[heads.num_attention_heads, heads.head_dim],
        ));
        specs.push(spec(n("attn.wk_dv.weight"), BF16, &[kv, h]));
        specs.push(spec(n("attn.wv_dv.weight"), BF16, &[kv, h]));
        for which in ["k", "v"] {
            match span {
                AttentionSpan::Sliding => specs.push(spec(
                    n(&format!("attn.{which}_sconv.weight")),
                    BF16,
                    &[kv, 1, kernel],
                )),
                AttentionSpan::Global => {
                    specs.push(spec(
                        n(&format!("attn.{which}_sconv.static_w")),
                        BF16,
                        &[kernel, kv],
                    ));
                    specs.push(spec(
                        n(&format!("attn.{which}_sconv.weight_proj")),
                        BF16,
                        &[heads.num_key_value_heads * kernel, h],
                    ));
                }
            }
        }
        specs.push(spec(n("attn.gate_attn.weight"), BF16, &[q, h]));
        specs.push(spec(n("attn.wo_ud.weight"), BF16, &[h, q]));
        specs.push(spec(
            n("attn.attn_sink_proj.weight"),
            BF16,
            &[heads.num_attention_heads, h],
        ));
        specs.push(spec(n("attn.wr_du.weight"), BF16, &[r, h]));
        specs.push(spec(
            n("attn.rel_l2_clip.scale"),
            BF16,
            &[heads.num_attention_heads, config.d_rel],
        ));
        specs.push(spec(
            n("attn.rel_logits_proj.proj"),
            BF16,
            &[config.d_rel, config.relative_extent(span)],
        ));
        match span {
            AttentionSpan::Sliding => specs.push(spec(n("attn_norm.weight"), BF16, &[h])),
            AttentionSpan::Global => {
                for which in ["q", "k", "v"] {
                    specs.push(spec(n(&format!("{which}_input_norm.weight")), BF16, &[h]));
                }
            }
        }
        specs.push(spec(n("attn_sconv.weight"), BF16, &[h, 1, kernel]));
        specs.push(spec(n("mlp_norm.weight"), BF16, &[h]));
        specs.push(spec(
            n("mlp.gate.weight"),
            BF16,
            &[config.n_routed_experts, h],
        ));
        specs.push(spec(n("mlp.gate.bias"), F32, &[config.n_routed_experts]));
        specs.extend(expert_specs(config, layer, format)?);
        specs.push(spec(n("mlp_sconv.weight"), BF16, &[h, 1, kernel]));
        if (lc.core_start..lc.core_end).contains(&layer) {
            for controller in ["attn", "k", "v", "mlp"] {
                let c = |part: &str| n(&format!("controllers.{controller}.{part}_weight"));
                specs.push(spec(c("down"), BF16, &[lc.gate_rank, 2 * h]));
                specs.push(spec(c("forget"), BF16, &[h, lc.gate_rank]));
                specs.push(spec(c("gain"), BF16, &[h, lc.gate_rank]));
            }
        }
        if layer > 0 {
            let shape: &[usize] = match span {
                AttentionSpan::Sliding => &[h],
                AttentionSpan::Global => &[3, h],
            };
            specs.push(spec(
                top(&format!("attn_res.attn_queries.{}", layer - 1)),
                F32,
                shape,
            ));
        }
        specs.push(spec(
            top(&format!("attn_res.mlp_queries.{layer}")),
            F32,
            &[h],
        ));
    }
    let core = lc.core_end - lc.core_start;
    specs.push(spec(top("attn_res.output_query"), F32, &[h]));
    specs.push(spec(top("embed.weight"), BF16, &[config.vocab_size, h]));
    specs.push(spec(top("embed_norm.weight"), BF16, &[h]));
    specs.push(spec(
        top("middle_cycle_cross_pass_queries"),
        F32,
        &[2 * core, h],
    ));
    specs.push(spec(
        top("middle_cycle_cross_pass_kv_queries"),
        F32,
        &[2 * core, h],
    ));
    specs.push(spec(top("norm.weight"), BF16, &[h]));
    specs.push(spec(top("unembed.weight"), BF16, &[config.vocab_size, h]));
    Ok(specs)
}

/// The safetensors files of a checkpoint, with the file that holds each tensor.
#[derive(Debug)]
pub struct Inkling2Checkpoint {
    files: Vec<SafetensorsFile>,
    location: HashMap<String, usize>,
}

impl Inkling2Checkpoint {
    /// Opens the weights in `dir`: the files `model.safetensors.index.json` names, or
    /// `model.safetensors` when there is no index.
    ///
    /// # Errors
    ///
    /// Returns the error of a file that cannot be opened or parsed, and
    /// [`InferenceError::InvalidSafetensors`] when the index places a tensor in a file that
    /// does not hold it or a tensor is stored in more than one file.
    pub fn open(dir: &Path) -> Result<Self, InferenceError> {
        if !dir.join("model.safetensors.index.json").is_file() {
            return Self::from_files(vec![SafetensorsFile::open(&dir.join("model.safetensors"))?]);
        }
        let index = parse_index(dir)?;
        let mut shards: Vec<&str> = index.weight_map.values().map(String::as_str).collect();
        shards.sort_unstable();
        shards.dedup();
        let mut files = Vec::with_capacity(shards.len());
        for shard in &shards {
            let (file, path) = open_manifest_entry_once(dir, shard)?;
            files.push(safetensors_from_open_file(file, &path)?);
        }
        let checkpoint = Self::from_files(files)?;
        for (name, shard) in &index.weight_map {
            let listed = shards.binary_search(&shard.as_str()).ok();
            if checkpoint.location.get(name).copied() != listed {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "the index places tensor {name} in {shard}, which does not hold it"
                )));
            }
        }
        Ok(checkpoint)
    }

    /// A checkpoint made of already parsed files.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidSafetensors`] when a tensor name appears in more than
    /// one file.
    pub fn from_files(files: Vec<SafetensorsFile>) -> Result<Self, InferenceError> {
        let mut location = HashMap::new();
        for (index, file) in files.iter().enumerate() {
            for name in file.tensor_names_iter() {
                if location.insert(name.to_string(), index).is_some() {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "tensor {name} is stored in more than one file"
                    )));
                }
            }
        }
        Ok(Self { files, location })
    }

    fn file(&self, name: &str) -> Result<&SafetensorsFile, InferenceError> {
        self.location
            .get(name)
            .and_then(|&index| self.files.get(index))
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    fn dtype(&self, name: &str) -> Result<&'static str, InferenceError> {
        self.file(name)?
            .tensor_dtype(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    fn shape(&self, name: &str) -> Result<&[usize], InferenceError> {
        self.file(name)?
            .tensor_shape(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    fn payload(&self, name: &str, dtype: &str) -> Result<&[u8], InferenceError> {
        self.file(name)?.raw_payload(name, dtype)
    }

    fn bf16_bits(&self, name: &str) -> Result<Vec<u16>, InferenceError> {
        let bytes = self.payload(name, BF16)?;
        let mut bits = Vec::with_capacity(bytes.len() / 2);
        for (index, pair) in bytes.chunks_exact(2).enumerate() {
            let value = u16::from_le_bytes([pair[0], pair[1]]);
            if value & 0x7f80 == 0x7f80 {
                return Err(bad_tensor(name, &format!("element {index} is not finite")));
            }
            bits.push(value);
        }
        Ok(bits)
    }

    fn bf16_values(&self, name: &str) -> Result<Vec<f32>, InferenceError> {
        Ok(self
            .bf16_bits(name)?
            .into_iter()
            .map(bf16_bits_to_f32)
            .collect())
    }

    fn bf16_matrix(&self, name: &str) -> Result<DenseMatrix, InferenceError> {
        let shape = self.shape(name)?;
        let rows = shape.first().copied().unwrap_or(0);
        let cols = shape.iter().skip(1).product();
        DenseMatrix::from_bf16_bits(rows, cols, self.bf16_bits(name)?)
            .map_err(|error| in_tensor(name, error))
    }

    fn f32_values(&self, name: &str) -> Result<Vec<f32>, InferenceError> {
        let bytes = self.payload(name, F32)?;
        let mut values = Vec::with_capacity(bytes.len() / 4);
        for (index, quad) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
            if !value.is_finite() {
                return Err(bad_tensor(name, &format!("element {index} is not finite")));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn bytes(&self, name: &str, dtype: &str) -> Result<Vec<u8>, InferenceError> {
        Ok(self.payload(name, dtype)?.to_vec())
    }

    fn expert_format(&self, layer: usize) -> Result<ExpertFormat, InferenceError> {
        let name = layer_tensor(layer, "mlp.experts.w13_weight");
        match self.dtype(&name)? {
            U8 => Ok(ExpertFormat::Nvfp4),
            F8_E4M3 => Ok(ExpertFormat::Mxfp8),
            other => Err(bad_tensor(
                &name,
                &format!(
                    "experts stored as {other} are read as neither NVFP4 (U8) nor MXFP8 (F8_E4M3)"
                ),
            )),
        }
    }

    fn check_inventory(&self, specs: &[TensorSpec]) -> Result<(), InferenceError> {
        for expected in specs {
            let dtype = self.dtype(&expected.name)?;
            if dtype != expected.dtype {
                return Err(bad_tensor(
                    &expected.name,
                    &format!("dtype {dtype}, expected {}", expected.dtype),
                ));
            }
            let shape = self.shape(&expected.name)?;
            if shape != expected.shape.as_slice() {
                return Err(bad_tensor(
                    &expected.name,
                    &format!("shape {shape:?}, expected {:?}", expected.shape),
                ));
            }
        }
        let read: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        let mut unread: Vec<&str> = self
            .location
            .keys()
            .map(String::as_str)
            .filter(|name| name.starts_with(TEXT_PREFIX) && !read.contains(name))
            .collect();
        unread.sort_unstable();
        match unread.first() {
            Some(first) => Err(bad_tensor(
                first,
                &format!(
                    "a text-decoder tensor this loader does not read ({} such tensors)",
                    unread.len()
                ),
            )),
            None => Ok(()),
        }
    }

    fn expert_bank(
        &self,
        layer: usize,
        which: &str,
        format: ExpertFormat,
        shape: (usize, usize, usize),
    ) -> Result<ExpertBank, InferenceError> {
        let (experts, rows, cols) = shape;
        let weight = layer_tensor(layer, &format!("mlp.experts.{which}_weight"));
        let scale = format!("{weight}_scale");
        match format {
            ExpertFormat::Nvfp4 => {
                let scales = self.bytes(&scale, F8_E4M3)?;
                if let Some(index) = scales.iter().position(|&s| s & 0x80 != 0) {
                    return Err(bad_tensor(
                        &scale,
                        &format!("scale {index} has its sign bit set"),
                    ));
                }
                let amax_name = format!("{weight}_amax");
                let denominator_name = format!("{weight}_outer_scale_denominator");
                let amax = self.f32_values(&amax_name)?;
                let denominator = self
                    .f32_values(&denominator_name)?
                    .first()
                    .copied()
                    .ok_or_else(|| bad_tensor(&denominator_name, "no value"))?;
                if denominator <= 0.0 {
                    return Err(bad_tensor(
                        &denominator_name,
                        &format!("denominator {denominator} is not positive"),
                    ));
                }
                let multipliers: Vec<f32> = amax.iter().map(|a| a / denominator).collect();
                if let Some(index) = multipliers
                    .iter()
                    .position(|m| !m.is_finite() || m.is_sign_negative())
                {
                    return Err(bad_tensor(
                        &amax_name,
                        &format!(
                            "expert {index} has amax {} over a denominator of {denominator}",
                            amax[index]
                        ),
                    ));
                }
                ExpertBank::nvfp4(
                    experts,
                    rows,
                    cols,
                    self.bytes(&weight, U8)?,
                    scales,
                    multipliers,
                )
                .map_err(|error| in_tensor(&weight, error))
            }
            ExpertFormat::Mxfp8 => {
                let scales = self.bytes(&scale, U8)?;
                if let Some(index) = scales.iter().position(|&s| s == u8::MAX) {
                    return Err(bad_tensor(
                        &scale,
                        &format!("scale {index} is 255, which encodes NaN"),
                    ));
                }
                ExpertBank::mxfp8(experts, rows, cols, self.bytes(&weight, F8_E4M3)?, scales)
                    .map_err(|error| in_tensor(&weight, error))
            }
        }
    }

    fn message_gate(
        &self,
        layer: usize,
        controller: &str,
    ) -> Result<MessageGateWeights, InferenceError> {
        let n =
            |part: &str| layer_tensor(layer, &format!("controllers.{controller}.{part}_weight"));
        Ok(MessageGateWeights {
            down: self.bf16_matrix(&n("down"))?,
            forget: self.bf16_matrix(&n("forget"))?,
            gain: self.bf16_matrix(&n("gain"))?,
        })
    }

    fn kv_convolution(
        &self,
        layer: usize,
        which: &str,
        span: AttentionSpan,
    ) -> Result<KvConvolution, InferenceError> {
        let n = |suffix: &str| layer_tensor(layer, &format!("attn.{which}_sconv.{suffix}"));
        Ok(match span {
            AttentionSpan::Sliding => KvConvolution::Static {
                taps: self.bf16_values(&n("weight"))?,
            },
            AttentionSpan::Global => KvConvolution::Dynamic {
                static_taps: self.bf16_values(&n("static_w"))?,
                projection: self.bf16_matrix(&n("weight_proj"))?,
            },
        })
    }

    fn layer(
        &self,
        config: &Inkling2TextConfig,
        layer: usize,
        format: ExpertFormat,
    ) -> Result<LayerWeights, InferenceError> {
        let h = config.hidden_size;
        let span = config
            .span(layer)
            .ok_or_else(|| invalid(format!("layer {layer} has no attention span")))?;
        let n = |suffix: &str| layer_tensor(layer, suffix);
        let attention_input_queries = if layer == 0 {
            Vec::new()
        } else {
            self.f32_values(&top(&format!("attn_res.attn_queries.{}", layer - 1)))?
                .chunks_exact(h)
                .map(<[f32]>::to_vec)
                .collect()
        };
        let attention_norms = match span {
            AttentionSpan::Sliding => vec![self.bf16_values(&n("attn_norm.weight"))?],
            AttentionSpan::Global => ["q", "k", "v"]
                .into_iter()
                .map(|which| self.bf16_values(&n(&format!("{which}_input_norm.weight"))))
                .collect::<Result<_, _>>()?,
        };
        let attention = AttentionWeights {
            query: self.bf16_matrix(&n("attn.wq_du.weight"))?,
            query_scale: self.bf16_values(&n("attn.q_l2_clip.scale"))?,
            key: self.bf16_matrix(&n("attn.wk_dv.weight"))?,
            value: self.bf16_matrix(&n("attn.wv_dv.weight"))?,
            key_convolution: self.kv_convolution(layer, "k", span)?,
            value_convolution: self.kv_convolution(layer, "v", span)?,
            output_gate: self.bf16_matrix(&n("attn.gate_attn.weight"))?,
            output: self.bf16_matrix(&n("attn.wo_ud.weight"))?,
            sink: self.bf16_matrix(&n("attn.attn_sink_proj.weight"))?,
            relative: self.bf16_matrix(&n("attn.wr_du.weight"))?,
            relative_scale: self.bf16_values(&n("attn.rel_l2_clip.scale"))?,
            relative_profiles: self.bf16_matrix(&n("attn.rel_logits_proj.proj"))?,
        };
        let experts = config.n_routed_experts;
        let intermediate = config.moe_intermediate_size;
        let moe = MoeWeights {
            router: self.bf16_matrix(&n("mlp.gate.weight"))?,
            selection_bias: self.f32_values(&n("mlp.gate.bias"))?,
            gate_up: self.expert_bank(layer, "w13", format, (experts, 2 * intermediate, h))?,
            down: self.expert_bank(layer, "w2", format, (experts, h, intermediate))?,
        };
        let lc = config.loop_config;
        let gates = if (lc.core_start..lc.core_end).contains(&layer) {
            Some(LoopGates {
                query: self.message_gate(layer, "attn")?,
                key: self.message_gate(layer, "k")?,
                value: self.message_gate(layer, "v")?,
                mlp: self.message_gate(layer, "mlp")?,
            })
        } else {
            None
        };
        Ok(LayerWeights {
            attention_input_queries,
            mlp_input_query: self.f32_values(&top(&format!("attn_res.mlp_queries.{layer}")))?,
            attention_norms,
            mlp_norm: self.bf16_values(&n("mlp_norm.weight"))?,
            attention,
            post_attention_convolution: self.bf16_values(&n("attn_sconv.weight"))?,
            post_mlp_convolution: self.bf16_values(&n("mlp_sconv.weight"))?,
            moe,
            gates,
        })
    }
}

impl Inkling2Weights {
    /// Reads the text decoder's weights for `config` from `checkpoint`. Each layer's expert
    /// format is taken from the stored dtype of its `w13_weight`. Tensors outside the text
    /// decoder are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::MissingTensor`] or [`InferenceError::InvalidSafetensors`]
    /// naming the tensor when one the decoder reads is absent or has another dtype or shape,
    /// when the checkpoint holds a text-decoder tensor the decoder does not read, and when a
    /// value is not finite, an NVFP4 scale is negative, an MXFP8 scale encodes NaN or an NVFP4
    /// multiplier is negative or not finite.
    pub fn load(
        checkpoint: &Inkling2Checkpoint,
        config: &Inkling2TextConfig,
    ) -> Result<Self, InferenceError> {
        let formats = (0..config.num_hidden_layers)
            .map(|layer| checkpoint.expert_format(layer))
            .collect::<Result<Vec<_>, _>>()?;
        checkpoint.check_inventory(&expected_text_tensors(config, &formats)?)?;

        let h = config.hidden_size;
        let core = config.loop_config.core_end - config.loop_config.core_start;
        let queries = checkpoint.f32_values(&top("middle_cycle_cross_pass_queries"))?;
        let kv_queries = checkpoint.f32_values(&top("middle_cycle_cross_pass_kv_queries"))?;
        let row = |values: &[f32], index: usize| values[index * h..(index + 1) * h].to_vec();
        let mut message_queries = Vec::with_capacity(4 * core);
        for j in 0..core {
            message_queries.push(row(&queries, 2 * j));
            message_queries.push(row(&kv_queries, 2 * j));
            message_queries.push(row(&kv_queries, 2 * j + 1));
            message_queries.push(row(&queries, 2 * j + 1));
        }
        let layers = formats
            .iter()
            .enumerate()
            .map(|(layer, &format)| checkpoint.layer(config, layer, format))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embedding: checkpoint.bf16_matrix(&top("embed.weight"))?,
            embedding_norm: checkpoint.bf16_values(&top("embed_norm.weight"))?,
            layers,
            message_queries,
            output_query: checkpoint.f32_values(&top("attn_res.output_query"))?,
            final_norm: checkpoint.bf16_values(&top("norm.weight"))?,
            unembedding: checkpoint.bf16_matrix(&top("unembed.weight"))?,
        })
    }
}

impl Inkling2Model {
    /// Loads the text decoder from a checkpoint directory holding `config.json` and the
    /// safetensors weights, as [`Inkling2Checkpoint::open`] finds them.
    ///
    /// # Errors
    ///
    /// Returns the configuration parser's error, the errors of [`Inkling2Checkpoint::open`]
    /// and [`Inkling2Weights::load`], and those of [`Inkling2Model::new`].
    pub fn open(dir: &Path) -> Result<Self, InferenceError> {
        let config = parse_config(&dir.join("config.json"))?;
        let weights = Inkling2Weights::load(&Inkling2Checkpoint::open(dir)?, &config.text)?;
        Self::new(config.text, weights)
    }
}
