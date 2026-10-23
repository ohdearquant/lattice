//! Chronos-2: an encoder-only forecasting model over patches of a scaled series, with attention
//! along time within each series and along the batch within each group of series (Ansari et al.,
//! "Chronos-2", 2025; reference implementation `amazon-science/chronos-forecasting`).
//!
//! The model directory holds the published `config.json` and `model.safetensors` (f32). One call
//! maps a batch of equal-length contexts, each optionally paired with known future covariate
//! values, to 21 quantile forecasts per series for up to `max_output_patches` patches. Series that
//! share a group id attend to each other; series in different groups never mix, so a batch of
//! distinct ids is a batch of independent univariate forecasts.
//!
//! Context preparation follows the reference model exactly: each series is standardised by its
//! own mean and population standard deviation over present values and passed through `asinh`,
//! missing values (`NaN`) are masked, and the series is cut into fixed patches aligned to its end.

use std::path::Path;

use serde_json::Value;

use crate::error::InferenceError;
use crate::forward::cpu::{matmul_bt, rms_norm};
use crate::weights::SafetensorsFile;

const SCALE_FLOOR: f32 = 1e-5;
const REG_TOKEN_ID: usize = 1;

/// The fields of the published `config.json` that determine the network.
#[derive(Debug, Clone, PartialEq)]
pub struct Chronos2Config {
    pub d_model: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub layer_norm_epsilon: f32,
    pub rope_theta: f32,
    /// Longest context the model reads; longer input keeps its most recent values.
    pub context_length: usize,
    /// Input and output patch length (equal, with stride equal to length).
    pub patch_size: usize,
    /// Most output patches one call produces.
    pub max_output_patches: usize,
    /// Quantile levels of the output head, in output order.
    pub quantiles: Vec<f32>,
    /// Divisor of the time index fed alongside each patch.
    pub time_encoding_scale: f32,
    pub use_arcsinh: bool,
    pub use_reg_token: bool,
}

fn field<'a>(obj: &'a Value, name: &str) -> Result<&'a Value, InferenceError> {
    obj.get(name)
        .filter(|v| !v.is_null())
        .ok_or_else(|| InferenceError::InvalidInput(format!("config.json: missing field {name}")))
}

fn usize_field(obj: &Value, name: &str) -> Result<usize, InferenceError> {
    field(obj, name)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "config.json: {name} is not a non-negative integer"
            ))
        })
}

fn f32_field(obj: &Value, name: &str) -> Result<f32, InferenceError> {
    field(obj, name)?
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| InferenceError::InvalidInput(format!("config.json: {name} is not a number")))
}

fn bool_field(obj: &Value, name: &str) -> Result<bool, InferenceError> {
    field(obj, name)?
        .as_bool()
        .ok_or_else(|| InferenceError::InvalidInput(format!("config.json: {name} is not a bool")))
}

impl Chronos2Config {
    /// Parses the published `config.json` and refuses configurations this implementation does not
    /// compute exactly.
    pub fn from_config_json(text: &str) -> Result<Self, InferenceError> {
        let cfg: Value = serde_json::from_str(text)
            .map_err(|e| InferenceError::InvalidInput(format!("config.json: {e}")))?;
        let chronos = field(&cfg, "chronos_config")?;
        let unsupported = |why: String| {
            Err(InferenceError::UnsupportedModel(format!(
                "Chronos-2: {why}"
            )))
        };

        let act = field(&cfg, "dense_act_fn")?.as_str().unwrap_or_default();
        if act != "relu" {
            return unsupported(format!("dense_act_fn {act:?} (only relu is implemented)"));
        }
        if cfg.get("is_gated_act").and_then(Value::as_bool) == Some(true) {
            return unsupported("gated activations".into());
        }
        let input_patch = usize_field(chronos, "input_patch_size")?;
        let input_stride = usize_field(chronos, "input_patch_stride")?;
        let output_patch = usize_field(chronos, "output_patch_size")?;
        if input_patch != input_stride || input_patch != output_patch || input_patch == 0 {
            return unsupported(format!(
                "patch size {input_patch}, stride {input_stride}, output patch {output_patch} \
                 (only equal, non-zero sizes are implemented)"
            ));
        }
        let quantiles = field(chronos, "quantiles")?
            .as_array()
            .ok_or_else(|| {
                InferenceError::InvalidInput("config.json: quantiles is not an array".into())
            })?
            .iter()
            .map(|q| {
                q.as_f64().map(|q| q as f32).ok_or_else(|| {
                    InferenceError::InvalidInput("config.json: quantile is not a number".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let context_length = usize_field(chronos, "context_length")?;
        let time_encoding_scale = match chronos.get("time_encoding_scale") {
            None | Some(Value::Null) => context_length as f32,
            Some(_) => f32_field(chronos, "time_encoding_scale")?,
        };

        let config = Self {
            d_model: usize_field(&cfg, "d_model")?,
            d_kv: usize_field(&cfg, "d_kv")?,
            num_heads: usize_field(&cfg, "num_heads")?,
            d_ff: usize_field(&cfg, "d_ff")?,
            num_layers: usize_field(&cfg, "num_layers")?,
            layer_norm_epsilon: f32_field(&cfg, "layer_norm_epsilon")?,
            rope_theta: f32_field(&cfg, "rope_theta")?,
            context_length,
            patch_size: input_patch,
            max_output_patches: usize_field(chronos, "max_output_patches")?,
            quantiles,
            time_encoding_scale,
            use_arcsinh: bool_field(chronos, "use_arcsinh")?,
            use_reg_token: bool_field(chronos, "use_reg_token")?,
        };
        if config.d_kv == 0 || !config.d_kv.is_multiple_of(2) || config.num_heads == 0 {
            return unsupported(format!(
                "d_kv {} / num_heads {} (RoPE needs an even, non-zero head width)",
                config.d_kv, config.num_heads
            ));
        }
        if config.quantiles.is_empty() || config.max_output_patches == 0 {
            return unsupported("no quantiles or no output patches".into());
        }
        if config.context_length == 0
            || config.time_encoding_scale.is_nan()
            || config.time_encoding_scale <= 0.0
        {
            return unsupported("context_length and time_encoding_scale must be positive".into());
        }
        Ok(config)
    }

    /// Steps of horizon one call covers.
    pub fn max_horizon(&self) -> usize {
        self.max_output_patches * self.patch_size
    }
}

/// `y = x Wᵀ (+ b)` over `rows` rows, with `W` stored `[out, in]`.
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

/// `output(relu(hidden(x))) + residual(x)`, all with bias, no normalisation.
#[derive(Debug, Clone)]
struct ResidualBlock {
    hidden: Linear,
    output: Linear,
    residual: Linear,
}

impl ResidualBlock {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = self.hidden.apply(x, rows);
        for v in &mut h {
            *v = v.max(0.0);
        }
        let mut y = self.output.apply(&h, rows);
        let r = self.residual.apply(x, rows);
        for (a, b) in y.iter_mut().zip(&r) {
            *a += b;
        }
        y
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

#[derive(Debug, Clone)]
struct Block {
    time_norm: Vec<f32>,
    time_attn: Attention,
    group_norm: Vec<f32>,
    group_attn: Attention,
    ff_norm: Vec<f32>,
    wi: Linear,
    wo: Linear,
}

struct Tensors<'a> {
    file: &'a SafetensorsFile,
    used: Vec<String>,
}

impl Tensors<'_> {
    fn get(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, InferenceError> {
        let (data, got) = self.file.get_f32_tensor(name)?;
        if got != shape {
            return Err(InferenceError::InvalidSafetensors(format!(
                "Chronos-2: tensor {name} has shape {got:?}, expected {shape:?}"
            )));
        }
        self.used.push(name.to_string());
        Ok(data.to_vec())
    }

    fn linear(
        &mut self,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
        bias: bool,
    ) -> Result<Linear, InferenceError> {
        let weight = self.get(&format!("{prefix}.weight"), &[out_dim, in_dim])?;
        let bias = if bias {
            Some(self.get(&format!("{prefix}.bias"), &[out_dim])?)
        } else {
            None
        };
        Ok(Linear {
            weight,
            bias,
            in_dim,
            out_dim,
        })
    }

    fn residual_block(
        &mut self,
        prefix: &str,
        in_dim: usize,
        h_dim: usize,
        out_dim: usize,
    ) -> Result<ResidualBlock, InferenceError> {
        Ok(ResidualBlock {
            hidden: self.linear(&format!("{prefix}.hidden_layer"), h_dim, in_dim, true)?,
            output: self.linear(&format!("{prefix}.output_layer"), out_dim, h_dim, true)?,
            residual: self.linear(&format!("{prefix}.residual_layer"), out_dim, in_dim, true)?,
        })
    }

    fn attention(
        &mut self,
        prefix: &str,
        d: usize,
        inner: usize,
    ) -> Result<Attention, InferenceError> {
        Ok(Attention {
            q: self.linear(&format!("{prefix}.q"), inner, d, false)?,
            k: self.linear(&format!("{prefix}.k"), inner, d, false)?,
            v: self.linear(&format!("{prefix}.v"), inner, d, false)?,
            o: self.linear(&format!("{prefix}.o"), d, inner, false)?,
        })
    }
}

/// One series of a call: its history and, optionally, known values over the forecast horizon.
#[derive(Debug, Clone, Default)]
pub struct Chronos2Series {
    /// History, oldest first; `NaN` marks a missing value.
    pub context: Vec<f32>,
    /// Series with equal ids attend to each other.
    pub group_id: u64,
    /// Known future values from the first forecast step on; `NaN` marks an unknown step. `None`
    /// is the same as all unknown: this series is a target to forecast.
    pub future_covariates: Option<Vec<f32>>,
}

/// Quantile forecasts of one series: `values[q * horizon + t]` is quantile `q` at step `t`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileForecast {
    pub horizon: usize,
    pub values: Vec<f32>,
}

impl QuantileForecast {
    /// The forecast at quantile index `q`.
    pub fn quantile(&self, q: usize) -> &[f32] {
        &self.values[q * self.horizon..(q + 1) * self.horizon]
    }
}

/// Encoder states recorded during a call, for comparison against a reference implementation.
#[derive(Debug, Clone, Default)]
pub struct Chronos2Trace {
    /// Encoder input embeddings, `[batch, positions, d_model]`.
    pub embeddings: Vec<f32>,
    /// Each block's output, same layout.
    pub blocks: Vec<Vec<f32>>,
    /// The final normalised state, same layout.
    pub final_state: Vec<f32>,
}

/// A loaded Chronos-2 model.
#[derive(Debug, Clone)]
pub struct Chronos2 {
    config: Chronos2Config,
    reg_embedding: Option<Vec<f32>>,
    input_embedding: ResidualBlock,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    output_embedding: ResidualBlock,
    inv_freq: Vec<f32>,
}

/// Mean and population standard deviation over present values, with the reference's fallbacks:
/// no present value gives location 0 and scale 1, and a zero spread gives the scale floor.
fn location_scale(x: &[f32]) -> (f32, f32) {
    let present: Vec<f32> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    if present.is_empty() {
        return (0.0, 1.0);
    }
    let n = present.len() as f32;
    let loc = present.iter().sum::<f32>() / n;
    let var = present.iter().map(|v| (v - loc) * (v - loc)).sum::<f32>() / n;
    let scale = var.sqrt();
    (loc, if scale == 0.0 { SCALE_FLOOR } else { scale })
}

fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

impl Chronos2 {
    /// Loads `config.json` and `model.safetensors` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, InferenceError> {
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let config = Chronos2Config::from_config_json(&text)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Self::from_safetensors(config, &file)
    }

    /// Builds the model from already-parsed tensors. Every tensor the configuration needs must be
    /// present with its exact shape, and no tensor may be left over.
    pub fn from_safetensors(
        config: Chronos2Config,
        file: &SafetensorsFile,
    ) -> Result<Self, InferenceError> {
        let mut t = Tensors {
            file,
            used: Vec::new(),
        };
        let (d, ff, p) = (config.d_model, config.d_ff, config.patch_size);
        let inner = config.num_heads * config.d_kv;
        let reg_embedding = if config.use_reg_token {
            let shared = t.get("shared.weight", &[2, d])?;
            Some(shared[REG_TOKEN_ID * d..(REG_TOKEN_ID + 1) * d].to_vec())
        } else {
            t.get("shared.weight", &[1, d])?;
            None
        };
        let input_embedding = t.residual_block("input_patch_embedding", 3 * p, ff, d)?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let b = format!("encoder.block.{i}.layer");
            blocks.push(Block {
                time_norm: t.get(&format!("{b}.0.layer_norm.weight"), &[d])?,
                time_attn: t.attention(&format!("{b}.0.self_attention"), d, inner)?,
                group_norm: t.get(&format!("{b}.1.layer_norm.weight"), &[d])?,
                group_attn: t.attention(&format!("{b}.1.self_attention"), d, inner)?,
                ff_norm: t.get(&format!("{b}.2.layer_norm.weight"), &[d])?,
                wi: t.linear(&format!("{b}.2.mlp.wi"), ff, d, false)?,
                wo: t.linear(&format!("{b}.2.mlp.wo"), d, ff, false)?,
            });
        }
        let final_norm = t.get("encoder.final_layer_norm.weight", &[d])?;
        let output_embedding =
            t.residual_block("output_patch_embedding", d, ff, config.quantiles.len() * p)?;

        let mut leftover: Vec<&str> = file
            .tensor_names()
            .into_iter()
            .filter(|name| !t.used.iter().any(|u| u == name))
            .collect();
        if !leftover.is_empty() {
            leftover.sort_unstable();
            return Err(InferenceError::InvalidSafetensors(format!(
                "Chronos-2: tensors not used by this configuration: {}",
                leftover.join(", ")
            )));
        }

        let half = config.d_kv / 2;
        let inv_freq = (0..half)
            .map(|i| 1.0 / config.rope_theta.powf((2 * i) as f32 / config.d_kv as f32))
            .collect();
        Ok(Self {
            config,
            reg_embedding,
            input_embedding,
            blocks,
            final_norm,
            output_embedding,
            inv_freq,
        })
    }

    pub fn config(&self) -> &Chronos2Config {
        &self.config
    }

    /// Forecasts `num_output_patches` patches for every series in `batch`.
    ///
    /// All contexts must have the same length (pad shorter ones on the left with `NaN`). Every
    /// series gets a forecast, including those whose future values were supplied as covariates.
    pub fn forward(
        &self,
        batch: &[Chronos2Series],
        num_output_patches: usize,
    ) -> Result<Vec<QuantileForecast>, InferenceError> {
        self.run(batch, num_output_patches, None)
    }

    /// [`forward`](Self::forward), also returning the encoder states.
    pub fn forward_with_trace(
        &self,
        batch: &[Chronos2Series],
        num_output_patches: usize,
    ) -> Result<(Vec<QuantileForecast>, Chronos2Trace), InferenceError> {
        let mut trace = Chronos2Trace::default();
        let out = self.run(batch, num_output_patches, Some(&mut trace))?;
        Ok((out, trace))
    }

    /// Forecasts one series on its own for `horizon` steps (at most [`Chronos2Config::max_horizon`]).
    pub fn predict(
        &self,
        context: &[f32],
        horizon: usize,
    ) -> Result<QuantileForecast, InferenceError> {
        if horizon == 0 || horizon > self.config.max_horizon() {
            return Err(InferenceError::InvalidInput(format!(
                "Chronos-2: horizon {horizon} is outside 1..={}",
                self.config.max_horizon()
            )));
        }
        let patches = horizon.div_ceil(self.config.patch_size);
        let series = Chronos2Series {
            context: context.to_vec(),
            ..Default::default()
        };
        let full = self
            .forward(std::slice::from_ref(&series), patches)?
            .remove(0);
        let values = (0..self.config.quantiles.len())
            .flat_map(|q| full.quantile(q)[..horizon].to_vec())
            .collect();
        Ok(QuantileForecast { horizon, values })
    }

    fn run(
        &self,
        batch: &[Chronos2Series],
        num_output_patches: usize,
        mut trace: Option<&mut Chronos2Trace>,
    ) -> Result<Vec<QuantileForecast>, InferenceError> {
        let cfg = &self.config;
        let (d, p) = (cfg.d_model, cfg.patch_size);
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        if num_output_patches == 0 || num_output_patches > cfg.max_output_patches {
            return Err(InferenceError::InvalidInput(format!(
                "Chronos-2: num_output_patches {num_output_patches} is outside 1..={}",
                cfg.max_output_patches
            )));
        }
        let raw_len = batch[0].context.len();
        if raw_len == 0 || batch.iter().any(|s| s.context.len() != raw_len) {
            return Err(InferenceError::InvalidInput(
                "Chronos-2: contexts must be non-empty and of equal length; pad on the left with NaN"
                    .into(),
            ));
        }
        let horizon = num_output_patches * p;
        for s in batch {
            if let Some(f) = &s.future_covariates
                && f.len() > horizon
            {
                return Err(InferenceError::InvalidInput(format!(
                    "Chronos-2: {} future covariate values exceed the {horizon}-step horizon",
                    f.len()
                )));
            }
        }

        let len = raw_len.min(cfg.context_length);
        let n_ctx = len.div_ceil(p);
        let pad = n_ctx * p - len;
        let reg = usize::from(cfg.use_reg_token);
        let seq = n_ctx + reg + num_output_patches;
        let bsz = batch.len();

        // Patch inputs [time(p), value(p), mask(p)] for every position that takes one, plus the
        // per-position attention mask and each series' location and scale.
        let mut ctx_rows = vec![0.0f32; bsz * n_ctx * 3 * p];
        let mut fut_rows = vec![0.0f32; bsz * num_output_patches * 3 * p];
        let mut key_mask = vec![1.0f32; bsz * seq];
        let mut loc_scale = Vec::with_capacity(bsz);
        let scale_value = |v: f32, loc: f32, scale: f32| {
            let z = (v - loc) / scale;
            if cfg.use_arcsinh { z.asinh() } else { z }
        };
        for (b, s) in batch.iter().enumerate() {
            let x = &s.context[raw_len - len..];
            let (loc, scale) = location_scale(x);
            loc_scale.push((loc, scale));
            for n in 0..n_ctx {
                let row = &mut ctx_rows[(b * n_ctx + n) * 3 * p..(b * n_ctx + n + 1) * 3 * p];
                let mut any = false;
                for j in 0..p {
                    let pos = n * p + j;
                    row[j] = (pos as f32 - (n_ctx * p) as f32) / cfg.time_encoding_scale;
                    if pos >= pad {
                        let v = x[pos - pad];
                        if !v.is_nan() {
                            row[p + j] = scale_value(v, loc, scale);
                            row[2 * p + j] = 1.0;
                            any = true;
                        }
                    }
                }
                if !any {
                    key_mask[b * seq + n] = 0.0;
                }
            }
            for n in 0..num_output_patches {
                let row = &mut fut_rows[(b * num_output_patches + n) * 3 * p
                    ..(b * num_output_patches + n + 1) * 3 * p];
                for j in 0..p {
                    let t = n * p + j;
                    row[j] = t as f32 / cfg.time_encoding_scale;
                    if let Some(f) = &s.future_covariates
                        && let Some(&v) = f.get(t)
                        && !v.is_nan()
                    {
                        row[p + j] = scale_value(v, loc, scale);
                        row[2 * p + j] = 1.0;
                    }
                }
            }
        }

        let ctx_embeds = self.input_embedding.apply(&ctx_rows, bsz * n_ctx);
        let fut_embeds = self
            .input_embedding
            .apply(&fut_rows, bsz * num_output_patches);
        let mut h = vec![0.0f32; bsz * seq * d];
        for b in 0..bsz {
            let dst = &mut h[b * seq * d..(b + 1) * seq * d];
            dst[..n_ctx * d].copy_from_slice(&ctx_embeds[b * n_ctx * d..(b + 1) * n_ctx * d]);
            if let Some(reg_embedding) = &self.reg_embedding {
                dst[n_ctx * d..(n_ctx + 1) * d].copy_from_slice(reg_embedding);
            }
            dst[(n_ctx + reg) * d..].copy_from_slice(
                &fut_embeds[b * num_output_patches * d..(b + 1) * num_output_patches * d],
            );
        }
        if let Some(t) = trace.as_deref_mut() {
            t.embeddings = h.clone();
        }

        let (cos, sin) = self.rope_tables(seq);
        let groups: Vec<u64> = batch.iter().map(|s| s.group_id).collect();
        for block in &self.blocks {
            self.time_attention(block, &mut h, bsz, seq, &key_mask, &cos, &sin);
            self.group_attention(block, &mut h, bsz, seq, &key_mask, &groups);
            self.feed_forward(block, &mut h, bsz * seq);
            if let Some(t) = trace.as_deref_mut() {
                t.blocks.push(h.clone());
            }
        }
        rms_norm(&mut h, &self.final_norm, d, cfg.layer_norm_epsilon);
        if let Some(t) = trace {
            t.final_state = h.clone();
        }

        let nq = cfg.quantiles.len();
        let mut out_rows = vec![0.0f32; bsz * num_output_patches * d];
        for b in 0..bsz {
            let src = &h[(b * seq + n_ctx + reg) * d..(b + 1) * seq * d];
            out_rows[b * num_output_patches * d..(b + 1) * num_output_patches * d]
                .copy_from_slice(src);
        }
        let y = self
            .output_embedding
            .apply(&out_rows, bsz * num_output_patches);
        let mut forecasts = Vec::with_capacity(bsz);
        for (b, &(loc, scale)) in loc_scale.iter().enumerate() {
            let mut values = vec![0.0f32; nq * horizon];
            for n in 0..num_output_patches {
                let row = &y[(b * num_output_patches + n) * nq * p..][..nq * p];
                for q in 0..nq {
                    for j in 0..p {
                        let mut v = row[q * p + j];
                        if cfg.use_arcsinh {
                            v = v.sinh();
                        }
                        values[q * horizon + n * p + j] = v * scale + loc;
                    }
                }
            }
            forecasts.push(QuantileForecast { horizon, values });
        }
        Ok(forecasts)
    }

    /// Rotary tables `[seq, d_kv]` in the rotate-half layout (each frequency repeated twice).
    fn rope_tables(&self, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let dk = self.config.d_kv;
        let half = dk / 2;
        let mut cos = vec![0.0f32; seq * dk];
        let mut sin = vec![0.0f32; seq * dk];
        for pos in 0..seq {
            for i in 0..half {
                let angle = pos as f32 * self.inv_freq[i];
                let (s, c) = angle.sin_cos();
                cos[pos * dk + i] = c;
                cos[pos * dk + half + i] = c;
                sin[pos * dk + i] = s;
                sin[pos * dk + half + i] = s;
            }
        }
        (cos, sin)
    }

    #[allow(clippy::too_many_arguments)]
    fn time_attention(
        &self,
        block: &Block,
        h: &mut [f32],
        bsz: usize,
        seq: usize,
        key_mask: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) {
        let cfg = &self.config;
        let (d, dk, nh) = (cfg.d_model, cfg.d_kv, cfg.num_heads);
        let inner = nh * dk;
        let half = dk / 2;
        let mut x = h.to_vec();
        rms_norm(&mut x, &block.time_norm, d, cfg.layer_norm_epsilon);
        let rows = bsz * seq;
        let mut q = block.time_attn.q.apply(&x, rows);
        let mut k = block.time_attn.k.apply(&x, rows);
        let v = block.time_attn.v.apply(&x, rows);
        for t in [&mut q, &mut k] {
            for r in 0..rows {
                let pos = r % seq;
                for head in 0..nh {
                    let a = &mut t[r * inner + head * dk..r * inner + (head + 1) * dk];
                    let (c, s) = (
                        &cos[pos * dk..(pos + 1) * dk],
                        &sin[pos * dk..(pos + 1) * dk],
                    );
                    let rotated: Vec<f32> = (0..dk)
                        .map(|i| if i < half { -a[i + half] } else { a[i - half] })
                        .collect();
                    for i in 0..dk {
                        a[i] = a[i] * c[i] + rotated[i] * s[i];
                    }
                }
            }
        }
        let mut attended = vec![0.0f32; rows * inner];
        let mut qh = vec![0.0f32; seq * dk];
        let mut kh = vec![0.0f32; seq * dk];
        let mut vh = vec![0.0f32; seq * dk];
        let mut scores = vec![0.0f32; seq * seq];
        for b in 0..bsz {
            for head in 0..nh {
                for s in 0..seq {
                    let src = (b * seq + s) * inner + head * dk;
                    qh[s * dk..(s + 1) * dk].copy_from_slice(&q[src..src + dk]);
                    kh[s * dk..(s + 1) * dk].copy_from_slice(&k[src..src + dk]);
                    vh[s * dk..(s + 1) * dk].copy_from_slice(&v[src..src + dk]);
                }
                matmul_bt(&qh, &kh, &mut scores, seq, dk, seq);
                for row in scores.chunks_exact_mut(seq) {
                    for (j, sc) in row.iter_mut().enumerate() {
                        *sc += (1.0 - key_mask[b * seq + j]) * f32::MIN;
                    }
                    softmax_row(row);
                }
                for s in 0..seq {
                    let dst = &mut attended[(b * seq + s) * inner + head * dk..][..dk];
                    for (j, &w) in scores[s * seq..(s + 1) * seq].iter().enumerate() {
                        for (o, &vv) in dst.iter_mut().zip(&vh[j * dk..(j + 1) * dk]) {
                            *o += w * vv;
                        }
                    }
                }
            }
        }
        let out = block.time_attn.o.apply(&attended, rows);
        for (a, o) in h.iter_mut().zip(&out) {
            *a += o;
        }
    }

    fn group_attention(
        &self,
        block: &Block,
        h: &mut [f32],
        bsz: usize,
        seq: usize,
        key_mask: &[f32],
        groups: &[u64],
    ) {
        let cfg = &self.config;
        let (d, dk, nh) = (cfg.d_model, cfg.d_kv, cfg.num_heads);
        let inner = nh * dk;
        let mut x = h.to_vec();
        rms_norm(&mut x, &block.group_norm, d, cfg.layer_norm_epsilon);
        let rows = bsz * seq;
        let q = block.group_attn.q.apply(&x, rows);
        let k = block.group_attn.k.apply(&x, rows);
        let v = block.group_attn.v.apply(&x, rows);
        let mut attended = vec![0.0f32; rows * inner];
        let mut scores = vec![0.0f32; bsz];
        for t in 0..seq {
            for head in 0..nh {
                for qb in 0..bsz {
                    let qv = &q[(qb * seq + t) * inner + head * dk..][..dk];
                    for (kb, sc) in scores.iter_mut().enumerate() {
                        let kv = &k[(kb * seq + t) * inner + head * dk..][..dk];
                        let dot: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum();
                        let allowed = if groups[qb] == groups[kb] {
                            key_mask[kb * seq + t]
                        } else {
                            0.0
                        };
                        *sc = dot + (1.0 - allowed) * f32::MIN;
                    }
                    softmax_row(&mut scores);
                    let dst = &mut attended[(qb * seq + t) * inner + head * dk..][..dk];
                    for (kb, &w) in scores.iter().enumerate() {
                        let vv = &v[(kb * seq + t) * inner + head * dk..][..dk];
                        for (o, &x) in dst.iter_mut().zip(vv) {
                            *o += w * x;
                        }
                    }
                }
            }
        }
        let out = block.group_attn.o.apply(&attended, rows);
        for (a, o) in h.iter_mut().zip(&out) {
            *a += o;
        }
    }

    fn feed_forward(&self, block: &Block, h: &mut [f32], rows: usize) {
        let cfg = &self.config;
        let mut x = h.to_vec();
        rms_norm(&mut x, &block.ff_norm, cfg.d_model, cfg.layer_norm_epsilon);
        let mut mid = block.wi.apply(&x, rows);
        for v in &mut mid {
            *v = v.max(0.0);
        }
        let out = block.wo.apply(&mid, rows);
        for (a, o) in h.iter_mut().zip(&out) {
            *a += o;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
        "d_model": 8, "d_kv": 4, "num_heads": 2, "d_ff": 16, "num_layers": 1,
        "layer_norm_epsilon": 1e-6, "rope_theta": 10000.0, "dense_act_fn": "relu",
        "is_gated_act": false,
        "chronos_config": {
            "context_length": 64, "input_patch_size": 4, "input_patch_stride": 4,
            "output_patch_size": 4, "max_output_patches": 3, "quantiles": [0.1, 0.5, 0.9],
            "use_arcsinh": true, "use_reg_token": true
        }
    }"#;

    #[test]
    fn config_parses_and_defaults_the_time_scale_to_the_context_length() {
        let c = Chronos2Config::from_config_json(CONFIG).unwrap();
        assert_eq!((c.d_model, c.d_kv, c.num_heads, c.patch_size), (8, 4, 2, 4));
        assert_eq!(c.time_encoding_scale, 64.0);
        assert_eq!(c.quantiles, vec![0.1, 0.5, 0.9]);
        assert_eq!(c.max_horizon(), 12);
    }

    #[test]
    fn config_refuses_what_is_not_implemented() {
        for (from, to) in [
            (r#""dense_act_fn": "relu""#, r#""dense_act_fn": "gelu""#),
            (r#""is_gated_act": false"#, r#""is_gated_act": true"#),
            (r#""input_patch_stride": 4"#, r#""input_patch_stride": 2"#),
            (r#""output_patch_size": 4"#, r#""output_patch_size": 8"#),
            (r#""d_kv": 4"#, r#""d_kv": 3"#),
        ] {
            let text = CONFIG.replace(from, to);
            assert_ne!(text, CONFIG, "fixture edit {to} did not apply");
            assert!(
                matches!(
                    Chronos2Config::from_config_json(&text),
                    Err(InferenceError::UnsupportedModel(_))
                ),
                "{to} was accepted"
            );
        }
    }

    #[test]
    fn location_and_scale_skip_missing_values_and_floor_a_flat_series() {
        let (loc, scale) = location_scale(&[1.0, f32::NAN, 3.0]);
        assert_eq!((loc, scale), (2.0, 1.0));
        assert_eq!(location_scale(&[5.0; 4]), (5.0, SCALE_FLOOR));
        assert_eq!(location_scale(&[f32::NAN; 3]), (0.0, 1.0));
    }

    #[test]
    fn a_fully_masked_softmax_row_stays_finite() {
        let mut row = [0.5 + f32::MIN, -0.25 + f32::MIN];
        softmax_row(&mut row);
        assert!(row.iter().all(|v| v.is_finite()));
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
