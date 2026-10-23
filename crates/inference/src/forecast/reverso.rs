//! Reverso: a univariate forecasting model that alternates long circular convolutions with
//! delta-rule linear attention, then reads a fixed block of future steps out through one
//! cross-attention head (Fu et al., "Reverso", 2025; reference implementation `shinfxh/reverso`).
//!
//! The model directory holds `args.json` (the reference training configuration) and
//! `model.safetensors` (f32 tensors under the reference state-dict names). Checkpoints are
//! converted to safetensors outside lattice; nothing here reads a pickle.
//!
//! One forward call maps a window of `seq_len` values to `output_token_len` future values.
//! [`Reverso::rollout`] repeats it autoregressively, and [`Reverso::predict`] wraps the rollout in
//! the context preparation the reference evaluation uses (imputation, padding, optional striding
//! and sign-flip averaging), so its output can be scored against published results.

use std::path::Path;

use serde_json::Value;

use super::fft::{Fft, FftScratch, KernelSpectrum};
use super::prep;
use crate::error::InferenceError;
use crate::forward::cpu::{layer_norm, matmul_bt, matmul_into};
use crate::weights::SafetensorsFile;

const NUM_HEADS: usize = 4;
const SHORT_CONV: usize = 4;
const LAYER_NORM_EPS: f32 = 1e-5;
const RMS_NORM_EPS: f32 = 1e-5;
const L2_NORM_EPS: f32 = 1e-6;
const RANGE_FLOOR: f32 = 1e-5;

/// The fields of the reference `args.json` that determine the network.
#[derive(Debug, Clone, PartialEq)]
pub struct ReversoConfig {
    pub seq_len: usize,
    pub output_token_len: usize,
    pub d_model: usize,
    pub d_intermediate: usize,
    /// One entry per mixing block, each followed by an MLP block.
    pub blocks: Vec<BlockKind>,
    pub use_norm: bool,
    pub head_bias: bool,
    pub value_dim: usize,
    pub state_weaving: bool,
    pub gating_kernel_size: usize,
    pub use_output_pe: bool,
}

/// The mixing block a layer position holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Gated long convolution over the whole window.
    Conv,
    /// Delta-rule linear attention.
    Attention,
}

fn field<'a>(args: &'a Value, name: &str) -> Result<&'a Value, InferenceError> {
    args.get(name)
        .filter(|v| !v.is_null())
        .ok_or_else(|| InferenceError::InvalidInput(format!("args.json: missing field {name}")))
}

fn usize_field(args: &Value, name: &str) -> Result<usize, InferenceError> {
    field(args, name)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!("args.json: {name} is not a non-negative integer"))
        })
}

/// Python truthiness of a bool or integer field; absent or null reads as `default`.
fn flag_field(args: &Value, name: &str, default: bool) -> Result<bool, InferenceError> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => v.as_i64().map(|i| i != 0).ok_or_else(|| {
            InferenceError::InvalidInput(format!("args.json: {name} is not a bool or integer"))
        }),
    }
}

impl ReversoConfig {
    /// Parses the reference `args.json` and refuses configurations this implementation does not
    /// compute exactly.
    pub fn from_args_json(text: &str) -> Result<Self, InferenceError> {
        let args: Value = serde_json::from_str(text)
            .map_err(|e| InferenceError::InvalidInput(format!("args.json: {e}")))?;
        let seq_len = usize_field(&args, "seq_len")?;
        let input_token_len = usize_field(&args, "input_token_len")?;
        let output_token_len = usize_field(&args, "output_token_len")?;
        let d_model = usize_field(&args, "d_model")?;
        let d_intermediate = usize_field(&args, "d_intermediate")?;
        let bottleneck = match args.get("output_bottleneck_dim") {
            None | Some(Value::Null) => output_token_len,
            Some(_) => usize_field(&args, "output_bottleneck_dim")?,
        };
        let expand_v = match args.get("expand_v") {
            None | Some(Value::Null) => 1.0,
            Some(v) => v.as_f64().ok_or_else(|| {
                InferenceError::InvalidInput("args.json: expand_v is not a number".into())
            })?,
        };
        let gating_kernel_size = match args.get("gating_kernel_size") {
            None | Some(Value::Null) => 3,
            Some(_) => usize_field(&args, "gating_kernel_size")?,
        };
        let main_module = field(&args, "main_module")?.as_str().ok_or_else(|| {
            InferenceError::InvalidInput("args.json: main_module is not a string".into())
        })?;
        let blocks = main_module
            .split(',')
            .map(|m| match m.trim() {
                "conv" => Ok(BlockKind::Conv),
                "attn" => Ok(BlockKind::Attention),
                other => Err(InferenceError::UnsupportedModel(format!(
                    "Reverso: unknown block type {other:?} in main_module"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let unsupported =
            |why: String| Err(InferenceError::UnsupportedModel(format!("Reverso: {why}")));
        if input_token_len != seq_len {
            return unsupported(format!(
                "input_token_len {input_token_len} differs from seq_len {seq_len}"
            ));
        }
        if bottleneck != output_token_len {
            return unsupported(format!(
                "output_bottleneck_dim {bottleneck} differs from output_token_len {output_token_len}"
            ));
        }
        if !seq_len.is_power_of_two() || seq_len < 2 {
            return unsupported(format!("seq_len {seq_len} is not a power of two"));
        }
        if output_token_len == 0 || d_model == 0 || !d_model.is_multiple_of(NUM_HEADS) {
            return unsupported(format!(
                "d_model {d_model} / output_token_len {output_token_len} not usable with {NUM_HEADS} heads"
            ));
        }
        let value_dim_f = d_model as f64 * expand_v;
        let value_dim = value_dim_f as usize;
        if value_dim as f64 != value_dim_f || value_dim == 0 || !value_dim.is_multiple_of(NUM_HEADS)
        {
            return unsupported(format!(
                "expand_v {expand_v} gives value width {value_dim_f}"
            ));
        }
        if gating_kernel_size.is_multiple_of(2) {
            return unsupported(format!("even gating_kernel_size {gating_kernel_size}"));
        }
        Ok(Self {
            seq_len,
            output_token_len,
            d_model,
            d_intermediate,
            blocks,
            use_norm: flag_field(&args, "use_norm", false)?,
            head_bias: flag_field(&args, "learn_bias", false)?,
            value_dim,
            state_weaving: flag_field(&args, "state_weaving", false)?,
            gating_kernel_size,
            use_output_pe: flag_field(&args, "use_output_pe", false)?,
        })
    }
}

/// `y = x Wᵀ + b` over `rows` rows, with `W` stored `[out, in]`.
#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * self.out_dim];
        matmul_bt(x, &self.weight, &mut y, rows, self.in_dim, self.out_dim);
        if let Some(bias) = &self.bias {
            for row in y.chunks_exact_mut(self.out_dim) {
                for (v, b) in row.iter_mut().zip(bias) {
                    *v += b;
                }
            }
        }
        y
    }
}

#[derive(Debug, Clone)]
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ConvBlock {
    kernels: Vec<KernelSpectrum>,
    gate_dw_weight: Vec<f32>,
    gate_dw_bias: Vec<f32>,
    gate_pw: Linear,
    norm: Norm,
}

#[derive(Debug, Clone)]
struct AttentionBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    beta: Linear,
    o_norm: Vec<f32>,
    o: Linear,
    norm: Norm,
    weave: bool,
}

#[derive(Debug, Clone)]
struct MlpBlock {
    up: Linear,
    down: Linear,
    norm: Norm,
}

#[derive(Debug, Clone)]
enum Layer {
    Conv(ConvBlock),
    Attention(Box<AttentionBlock>),
    Mlp(MlpBlock),
}

#[derive(Debug, Clone)]
struct OutputPositions {
    table: Vec<f32>,
    q_proj: Linear,
}

/// Options for [`Reverso::predict`], matching the reference evaluation's knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictOptions {
    /// Keep every `downsample`-th context value, forecast `horizon / downsample` steps, then
    /// resample linearly to `horizon`. 1 disables it.
    pub downsample: usize,
    /// Average the forecast with the negated forecast of the negated context:
    /// `0.5 (f(x) - f(-x))`.
    pub flip_invariance: bool,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            downsample: 1,
            flip_invariance: false,
        }
    }
}

/// A loaded Reverso model.
#[derive(Debug, Clone)]
pub struct Reverso {
    config: ReversoConfig,
    embedding: Vec<f32>,
    layers: Vec<Layer>,
    head: Linear,
    simple_q: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    output_positions: Option<OutputPositions>,
    fft: Fft,
}

struct Tensors<'a> {
    file: &'a SafetensorsFile,
}

impl Tensors<'_> {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, InferenceError> {
        let (data, got) = self.file.get_f32_tensor(name)?;
        if got != shape {
            return Err(InferenceError::InvalidSafetensors(format!(
                "Reverso: tensor {name} has shape {got:?}, expected {shape:?}"
            )));
        }
        Ok(data.to_vec())
    }

    fn linear(
        &self,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
        bias: bool,
    ) -> Result<Linear, InferenceError> {
        Ok(Linear {
            weight: self.get(&format!("{prefix}.weight"), &[out_dim, in_dim])?,
            bias: if bias {
                Some(self.get(&format!("{prefix}.bias"), &[out_dim])?)
            } else {
                None
            },
            in_dim,
            out_dim,
        })
    }

    fn norm(&self, prefix: &str, dim: usize) -> Result<Norm, InferenceError> {
        Ok(Norm {
            weight: self.get(&format!("{prefix}.weight"), &[dim])?,
            bias: self.get(&format!("{prefix}.bias"), &[dim])?,
        })
    }
}

/// Round to the nearest bfloat16 (ties to even), returned widened to f32.
fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Sinusoidal position table `[n, d]`: even columns `sin(p·w_i)`, odd columns `cos(p·w_i)`,
/// `w_i = exp(-2i·ln(10000)/d)`.
fn position_table(n: usize, d: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; n * d];
    let scale = -(10000.0f32.ln() / d as f32);
    for p in 0..n {
        for i in (0..d).step_by(2) {
            let angle = p as f32 * (i as f32 * scale).exp();
            table[p * d + i] = angle.sin();
            if i + 1 < d {
                table[p * d + i + 1] = angle.cos();
            }
        }
    }
    table
}

/// Depthwise convolution along time of a `[len, channels]` matrix with `taps` taps per channel
/// (`weight[c * taps + j]`), reading `x[t + j - offset]` and treating out-of-range rows as zero.
fn depthwise_conv(
    x: &[f32],
    len: usize,
    channels: usize,
    weight: &[f32],
    taps: usize,
    offset: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; len * channels];
    for t in 0..len {
        let row = &mut y[t * channels..(t + 1) * channels];
        for j in 0..taps {
            let Some(src) = (t + j).checked_sub(offset).filter(|&s| s < len) else {
                continue;
            };
            let src_row = &x[src * channels..(src + 1) * channels];
            for ((r, w), s) in row
                .iter_mut()
                .zip(weight[j..].iter().step_by(taps))
                .zip(src_row)
            {
                *r += w * s;
            }
        }
    }
    y
}

impl ConvBlock {
    fn forward(
        &self,
        h: &mut [f32],
        len: usize,
        d: usize,
        taps: usize,
        fft: &Fft,
        scratch: &mut FftScratch,
    ) {
        let x_bf16: Vec<f32> = h.iter().map(|&v| round_bf16(v)).collect();
        let mut gate = depthwise_conv(&x_bf16, len, d, &self.gate_dw_weight, taps, taps / 2);
        for row in gate.chunks_exact_mut(d) {
            for (v, b) in row.iter_mut().zip(&self.gate_dw_bias) {
                *v = silu(*v + b);
            }
        }
        let gate = self.gate_pw.apply(&gate, len);

        let mut column = vec![0.0f32; len];
        let mut conv_out = vec![0.0f32; len];
        let mut mixed = vec![0.0f32; len * d];
        for c in 0..d {
            for (t, col) in column.iter_mut().enumerate() {
                let i = t * d + c;
                *col = round_bf16(round_bf16(sigmoid(gate[i])) * x_bf16[i]);
            }
            fft.circular_convolve(&column, &self.kernels[c], &mut conv_out, scratch);
            for (t, &y) in conv_out.iter().enumerate() {
                mixed[t * d + c] = y.max(0.0);
            }
        }
        layer_norm(
            &mut mixed,
            &self.norm.weight,
            &self.norm.bias,
            d,
            LAYER_NORM_EPS,
        );
        for (v, m) in h.iter_mut().zip(&mixed) {
            *v += m;
        }
    }
}

impl AttentionBlock {
    fn forward(&self, h: &mut [f32], len: usize, d: usize, value_dim: usize) {
        let mut input = h.to_vec();
        if self.weave && len > 1 {
            for c in 0..d {
                input[c] += input[(len - 1) * d + c];
            }
        }
        let short_conv = |lin: &Linear, weight: &[f32]| {
            let mut y = depthwise_conv(
                &lin.apply(&input, len),
                len,
                lin.out_dim,
                weight,
                SHORT_CONV,
                SHORT_CONV - 1,
            );
            for v in y.iter_mut() {
                *v = silu(*v);
            }
            y
        };
        let mut q = short_conv(&self.q, &self.q_conv);
        let mut k = short_conv(&self.k, &self.k_conv);
        let v = short_conv(&self.v, &self.v_conv);
        let beta: Vec<f32> = self
            .beta
            .apply(&input, len)
            .into_iter()
            .map(sigmoid)
            .collect();

        let hk = d / NUM_HEADS;
        let hv = value_dim / NUM_HEADS;
        let q_scale = (hk as f32).powf(-0.5);
        for (mat, scale) in [(&mut q, q_scale), (&mut k, 1.0)] {
            for head in mat.chunks_exact_mut(hk) {
                let norm = (head.iter().map(|x| x * x).sum::<f32>() + L2_NORM_EPS).sqrt();
                for x in head.iter_mut() {
                    *x = *x / norm * scale;
                }
            }
        }

        let mut o = vec![0.0f32; len * value_dim];
        let mut state = vec![0.0f32; hk * hv];
        let mut retrieved = vec![0.0f32; hv];
        for head in 0..NUM_HEADS {
            state.fill(0.0);
            for t in 0..len {
                let kt = &k[t * d + head * hk..t * d + (head + 1) * hk];
                let qt = &q[t * d + head * hk..t * d + (head + 1) * hk];
                let vt = &v[t * value_dim + head * hv..t * value_dim + (head + 1) * hv];
                let b = beta[t * NUM_HEADS + head];
                retrieved.fill(0.0);
                for (ki, row) in kt.iter().zip(state.chunks_exact(hv)) {
                    for (r, s) in retrieved.iter_mut().zip(row) {
                        *r += ki * s;
                    }
                }
                for (ki, row) in kt.iter().zip(state.chunks_exact_mut(hv)) {
                    let w = b * ki;
                    for ((s, vj), r) in row.iter_mut().zip(vt).zip(&retrieved) {
                        *s += w * (vj - r);
                    }
                }
                let ot = &mut o[t * value_dim + head * hv..t * value_dim + (head + 1) * hv];
                for (qi, row) in qt.iter().zip(state.chunks_exact(hv)) {
                    for (oj, s) in ot.iter_mut().zip(row) {
                        *oj += qi * s;
                    }
                }
            }
        }
        for head in o.chunks_exact_mut(hv) {
            let ms = head.iter().map(|x| x * x).sum::<f32>() / hv as f32;
            let r = 1.0 / (ms + RMS_NORM_EPS).sqrt();
            for (x, w) in head.iter_mut().zip(&self.o_norm) {
                *x = *x * r * w;
            }
        }
        let mut out = self.o.apply(&o, len);
        layer_norm(
            &mut out,
            &self.norm.weight,
            &self.norm.bias,
            d,
            LAYER_NORM_EPS,
        );
        for (x, y) in h.iter_mut().zip(&out) {
            *x += y;
        }
    }
}

impl MlpBlock {
    fn forward(&self, h: &mut [f32], len: usize, d: usize) {
        let mut up = self.up.apply(h, len);
        for v in up.iter_mut() {
            *v = v.max(0.0);
        }
        let mut down = self.down.apply(&up, len);
        layer_norm(
            &mut down,
            &self.norm.weight,
            &self.norm.bias,
            d,
            LAYER_NORM_EPS,
        );
        for (x, y) in h.iter_mut().zip(&down) {
            *x += y;
        }
    }
}

impl Reverso {
    /// Loads `args.json` and `model.safetensors` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, InferenceError> {
        let args = std::fs::read_to_string(dir.join("args.json"))?;
        let config = ReversoConfig::from_args_json(&args)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Self::from_safetensors(config, &file)
    }

    /// Builds the model from already-parsed tensors. Every tensor the configuration needs must
    /// be present with its exact shape, and no tensor may be left over.
    pub fn from_safetensors(
        config: ReversoConfig,
        file: &SafetensorsFile,
    ) -> Result<Self, InferenceError> {
        let t = Tensors { file };
        let (d, l, dv) = (config.d_model, config.seq_len, config.value_dim);
        let taps = config.gating_kernel_size;
        let fft = Fft::new(l)?;
        let mut used: Vec<String> = Vec::new();
        let mut layers = Vec::with_capacity(config.blocks.len() * 2);
        let last = config.blocks.len().saturating_sub(1);
        for (i, kind) in config.blocks.iter().enumerate() {
            let p = format!("layers.{}", 2 * i);
            match kind {
                BlockKind::Conv => {
                    let k = t.get(&format!("{p}.k"), &[d, l])?;
                    layers.push(Layer::Conv(ConvBlock {
                        kernels: k
                            .chunks_exact(l)
                            .map(|row| fft.kernel_spectrum(row))
                            .collect(),
                        gate_dw_weight: t
                            .get(&format!("{p}.pregate.net.0.weight"), &[d, 1, taps])?,
                        gate_dw_bias: t.get(&format!("{p}.pregate.net.0.bias"), &[d])?,
                        gate_pw: Linear {
                            weight: t.get(&format!("{p}.pregate.net.2.weight"), &[d, d, 1])?,
                            bias: Some(t.get(&format!("{p}.pregate.net.2.bias"), &[d])?),
                            in_dim: d,
                            out_dim: d,
                        },
                        norm: t.norm(&format!("{p}.norm"), d)?,
                    }));
                    used.extend(
                        [
                            "k",
                            "pregate.net.0.weight",
                            "pregate.net.0.bias",
                            "pregate.net.2.weight",
                            "pregate.net.2.bias",
                            "norm.weight",
                            "norm.bias",
                        ]
                        .iter()
                        .map(|s| format!("{p}.{s}")),
                    );
                }
                BlockKind::Attention => {
                    let a = format!("{p}.attention");
                    layers.push(Layer::Attention(Box::new(AttentionBlock {
                        q: t.linear(&format!("{a}.q_proj"), d, d, false)?,
                        k: t.linear(&format!("{a}.k_proj"), d, d, false)?,
                        v: t.linear(&format!("{a}.v_proj"), dv, d, false)?,
                        q_conv: t.get(&format!("{a}.q_conv1d.weight"), &[d, 1, SHORT_CONV])?,
                        k_conv: t.get(&format!("{a}.k_conv1d.weight"), &[d, 1, SHORT_CONV])?,
                        v_conv: t.get(&format!("{a}.v_conv1d.weight"), &[dv, 1, SHORT_CONV])?,
                        beta: t.linear(&format!("{a}.b_proj"), NUM_HEADS, d, false)?,
                        o_norm: t.get(&format!("{a}.o_norm.weight"), &[dv / NUM_HEADS])?,
                        o: t.linear(&format!("{a}.o_proj"), d, dv, false)?,
                        norm: t.norm(&format!("{p}.norm"), d)?,
                        weave: config.state_weaving && i > 0 && i < last,
                    })));
                    used.extend(
                        [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "q_conv1d.weight",
                            "k_conv1d.weight",
                            "v_conv1d.weight",
                            "b_proj.weight",
                            "o_norm.weight",
                            "o_proj.weight",
                        ]
                        .iter()
                        .map(|s| format!("{a}.{s}")),
                    );
                    used.extend(
                        ["norm.weight", "norm.bias"]
                            .iter()
                            .map(|s| format!("{p}.{s}")),
                    );
                }
            }
            let m = format!("layers.{}", 2 * i + 1);
            let di = config.d_intermediate;
            let mlp = if di > 0 {
                MlpBlock {
                    up: t.linear(&format!("{m}.linear"), di, d, true)?,
                    down: t.linear(&format!("{m}.linear_final"), d, di, true)?,
                    norm: t.norm(&format!("{m}.norm"), d)?,
                }
            } else {
                return Err(InferenceError::UnsupportedModel(
                    "Reverso: d_intermediate 0 (single-linear MLP) is not implemented".into(),
                ));
            };
            layers.push(Layer::Mlp(mlp));
            used.extend(
                [
                    "linear.weight",
                    "linear.bias",
                    "linear_final.weight",
                    "linear_final.bias",
                    "norm.weight",
                    "norm.bias",
                ]
                .iter()
                .map(|s| format!("{m}.{s}")),
            );
        }

        let ol = config.output_token_len;
        let head = t.linear("head", ol, l, config.head_bias)?;
        let simple_q = t.linear("simple_q_proj", d, d, true)?;
        let key = t.linear("key_proj", d, d, true)?;
        let value = t.linear("value_proj", d, d, true)?;
        let out = t.linear("out_proj", 1, d, true)?;
        used.push("embedding.weight".into());
        used.push("head.weight".into());
        if config.head_bias {
            used.push("head.bias".into());
        }
        for name in ["simple_q_proj", "key_proj", "value_proj", "out_proj"] {
            used.push(format!("{name}.weight"));
            used.push(format!("{name}.bias"));
        }
        let output_positions = if config.use_output_pe {
            let n = l + ol;
            let table = if file.has_tensor("output_position_embedding.pe") {
                used.push("output_position_embedding.pe".into());
                t.get("output_position_embedding.pe", &[1, n, d])?
            } else {
                position_table(n, d)
            };
            used.push("post_pe_q_proj.weight".into());
            used.push("post_pe_q_proj.bias".into());
            Some(OutputPositions {
                table,
                q_proj: t.linear("post_pe_q_proj", d, d, true)?,
            })
        } else {
            None
        };

        let mut leftover: Vec<&str> = file
            .tensor_names()
            .into_iter()
            .filter(|name| !used.iter().any(|u| u == name))
            .collect();
        if !leftover.is_empty() {
            leftover.sort_unstable();
            return Err(InferenceError::InvalidSafetensors(format!(
                "Reverso: tensors not used by this configuration: {}",
                leftover.join(", ")
            )));
        }

        Ok(Self {
            embedding: t.get("embedding.weight", &[d, 1])?,
            layers,
            head,
            simple_q,
            key,
            value,
            out,
            output_positions,
            fft,
            config,
        })
    }

    pub fn config(&self) -> &ReversoConfig {
        &self.config
    }

    /// One forward call: `seq_len` finite values in, `output_token_len` values out.
    pub fn forward(&self, window: &[f32]) -> Result<Vec<f32>, InferenceError> {
        self.forward_inner(window, None)
    }

    /// [`Reverso::forward`], also returning each layer's output as a `[seq_len, d_model]`
    /// row-major matrix, in layer order (mixing and MLP blocks interleaved).
    #[doc(hidden)]
    pub fn forward_with_layer_outputs(
        &self,
        window: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), InferenceError> {
        let mut layers = Vec::with_capacity(self.layers.len());
        let y = self.forward_inner(window, Some(&mut layers))?;
        Ok((y, layers))
    }

    fn forward_inner(
        &self,
        window: &[f32],
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.config;
        let (l, d) = (cfg.seq_len, cfg.d_model);
        if window.len() != l {
            return Err(InferenceError::InvalidInput(format!(
                "Reverso: window has {} values, the model takes {l}",
                window.len()
            )));
        }
        if let Some(i) = window.iter().position(|v| !v.is_finite()) {
            return Err(InferenceError::InvalidInput(format!(
                "Reverso: window value {i} is not finite; impute missing values first"
            )));
        }
        let (offset, scale) = if cfg.use_norm {
            let lo = window.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (lo, (hi - lo).max(RANGE_FLOOR))
        } else {
            (0.0, 1.0)
        };

        let mut h = vec![0.0f32; l * d];
        for (t, &x) in window.iter().enumerate() {
            let xn = if cfg.use_norm {
                (x - offset) / scale
            } else {
                x
            };
            for (dst, w) in h[t * d..(t + 1) * d].iter_mut().zip(&self.embedding) {
                *dst = xn * w;
            }
        }
        let mut scratch = self.fft.scratch();
        for layer in &self.layers {
            match layer {
                Layer::Conv(b) => b.forward(
                    &mut h,
                    l,
                    d,
                    cfg.gating_kernel_size,
                    &self.fft,
                    &mut scratch,
                ),
                Layer::Attention(b) => b.forward(&mut h, l, d, cfg.value_dim),
                Layer::Mlp(b) => b.forward(&mut h, l, d),
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(h.clone());
            }
        }

        let ol = cfg.output_token_len;
        let mut summary = vec![0.0f32; ol * d];
        matmul_into(&self.head.weight, &h, &mut summary, ol, l, d);
        if let Some(bias) = &self.head.bias {
            for (row, b) in summary.chunks_exact_mut(d).zip(bias) {
                for v in row.iter_mut() {
                    *v += b;
                }
            }
        }
        let mut q = self.simple_q.apply(&summary, ol);
        if let Some(pe) = &self.output_positions {
            for (v, p) in h.iter_mut().zip(&pe.table[..l * d]) {
                *v += p;
            }
            for (v, p) in q.iter_mut().zip(&pe.table[l * d..(l + ol) * d]) {
                *v += p;
            }
            q = pe.q_proj.apply(&q, ol);
        }
        let k = self.key.apply(&h, l);
        let v = self.value.apply(&h, l);

        let mut scores = vec![0.0f32; ol * l];
        matmul_bt(&q, &k, &mut scores, ol, d, l);
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        for row in scores.chunks_exact_mut(l) {
            let mut max = f32::NEG_INFINITY;
            for s in row.iter_mut() {
                *s *= inv_sqrt_d;
                max = max.max(*s);
            }
            let mut sum = 0.0f32;
            for s in row.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in row.iter_mut() {
                *s /= sum;
            }
        }
        let mut attended = vec![0.0f32; ol * d];
        matmul_into(&scores, &v, &mut attended, ol, l, d);
        let y = self.out.apply(&attended, ol);
        Ok(y.into_iter().map(|v| v * scale + offset).collect())
    }

    /// Autoregressive forecast of `horizon` steps from a finite context of at least `seq_len`
    /// values: each call forecasts `output_token_len` steps from the latest `seq_len` values,
    /// which then join the context.
    pub fn rollout(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>, InferenceError> {
        let (l, ol) = (self.config.seq_len, self.config.output_token_len);
        if context.len() < l {
            return Err(InferenceError::InvalidInput(format!(
                "Reverso: context has {} values, rollout needs at least {l}",
                context.len()
            )));
        }
        let steps = horizon.div_ceil(ol);
        let mut series = context[context.len() - l..].to_vec();
        series.reserve(steps * ol);
        for _ in 0..steps {
            let chunk = self.forward(&series[series.len() - l..])?;
            series.extend_from_slice(&chunk);
        }
        Ok(series[l..l + horizon].to_vec())
    }

    /// The reference evaluation's prediction for one raw series: stride by `downsample`, pad or
    /// truncate to `seq_len`, impute missing values, roll out, optionally average with the sign
    /// flip, fill any non-finite step forward, and resample to `horizon` when downsampled.
    pub fn predict(
        &self,
        raw: &[f32],
        horizon: usize,
        options: PredictOptions,
    ) -> Result<Vec<f32>, InferenceError> {
        let factor = options.downsample.max(1);
        let target = horizon / factor;
        if target == 0 {
            return Err(InferenceError::InvalidInput(format!(
                "Reverso: horizon {horizon} is shorter than the downsample factor {factor}"
            )));
        }
        let mut context = prep::left_pad(&prep::stride(raw, factor), self.config.seq_len);
        prep::impute_missing(&mut context);
        let mut forecast = self.rollout(&context, target)?;
        if options.flip_invariance {
            let negated: Vec<f32> = context.iter().map(|v| -v).collect();
            let flipped = self.rollout(&negated, target)?;
            for (f, g) in forecast.iter_mut().zip(&flipped) {
                *f = 0.5 * (*f - g);
            }
        }
        if forecast.iter().any(|v| v.is_nan()) {
            prep::forward_fill(&mut forecast);
        }
        if factor > 1 {
            forecast = prep::resample_linear(&forecast, horizon);
        }
        Ok(forecast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NANO_ARGS: &str = r#"{"seq_len": 2048, "input_token_len": 2048, "output_token_len": 48,
        "d_model": 32, "d_intermediate": 128, "main_module": "conv,attn", "use_norm": true,
        "learn_bias": 1, "output_bottleneck_dim": 48, "expand_v": 1.0, "state_weaving": 1,
        "gating_kernel_size": 3, "use_output_pe": null}"#;

    #[test]
    fn parses_the_reference_args() {
        let c = ReversoConfig::from_args_json(NANO_ARGS).unwrap();
        assert_eq!(c.blocks, vec![BlockKind::Conv, BlockKind::Attention]);
        assert!(c.use_norm && c.head_bias && c.state_weaving && !c.use_output_pe);
        assert_eq!(
            (c.seq_len, c.output_token_len, c.d_model, c.value_dim),
            (2048, 48, 32, 32)
        );
    }

    #[test]
    fn refuses_configurations_it_does_not_compute() {
        for (from, to) in [
            ("\"seq_len\": 2048", "\"seq_len\": 1000"),
            ("\"input_token_len\": 2048", "\"input_token_len\": 1024"),
            (
                "\"main_module\": \"conv,attn\"",
                "\"main_module\": \"conv,mamba\"",
            ),
            ("\"gating_kernel_size\": 3", "\"gating_kernel_size\": 4"),
            ("\"d_model\": 32", "\"d_model\": 30"),
        ] {
            let text = NANO_ARGS.replace(from, to);
            assert_ne!(text, NANO_ARGS, "fixture edit {from} did not apply");
            assert!(
                ReversoConfig::from_args_json(&text).is_err(),
                "{to} must be refused"
            );
        }
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        assert_eq!(round_bf16(1.0), 1.0);
        let half_ulp = f32::from_bits(0x3f80_8000);
        assert_eq!(round_bf16(half_ulp), 1.0);
        let above = f32::from_bits(0x3f80_8001);
        assert_eq!(round_bf16(above).to_bits(), 0x3f81_0000);
        let odd_tie = f32::from_bits(0x3f81_8000);
        assert_eq!(round_bf16(odd_tie).to_bits(), 0x3f82_0000);
        assert!(round_bf16(f32::NAN).is_nan());
        assert_eq!(round_bf16(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn position_table_alternates_sine_and_cosine() {
        let t = position_table(3, 4);
        assert_eq!(&t[0..4], &[0.0, 1.0, 0.0, 1.0]);
        let w1 = (-(2.0 * 10000.0f32.ln()) / 4.0).exp();
        assert!((t[2 * 4 + 2] - (2.0 * w1).sin()).abs() < 1e-6);
        assert!((t[2 * 4 + 3] - (2.0 * w1).cos()).abs() < 1e-6);
    }
}
