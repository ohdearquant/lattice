//! Toto 2.0: a decoder-only forecasting model over patches of a causally scaled series, with
//! attention along time within each variate and across the variates of one series (Khwaja et
//! al., "Toto 2.0: Time Series Forecasting Enters the Scaling Era", 2026; reference
//! implementation `DataDog/toto`, `toto2/toto2/model.py`).
//!
//! The model directory holds the published `config.json` and `model.safetensors` (f32). The
//! weights were trained under unit scaling (u-μP), so every projection scales its output by a
//! fixed function of its input width; those factors are part of the network, not of the weights.
//!
//! One call forecasts every variate of a set of equal-length contexts, by default in a single
//! forward pass, the configuration the reference evaluates with; block decoding with a key/value
//! cache is available through [`Toto2Options::decode_block_size`]. Variates that share a series
//! id attend to each other; different ids never mix.

use std::path::Path;

use serde::Deserialize;

use crate::error::InferenceError;
use crate::forecast::QuantileForecast;
use crate::forward::cpu::{matmul_bt, rms_norm};
use crate::weights::SafetensorsFile;

const QUANTILES: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const MIN_SCALE: f64 = 1e-6;
const ROPE_BASE: f32 = 10_000.0;
const XPOS_SCALE_BASE: f32 = 256.0;
/// Positions the reference's rotary tables cover.
const MAX_POSITIONS: usize = 8192;

/// The published `config.json`, with the reference's defaults applied.
#[derive(Debug, Clone, PartialEq)]
pub struct Toto2Config {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_group_size: usize,
    pub num_variate_layers_per_group: usize,
    pub variate_layer_first: bool,
    pub norm_eps: f32,
    pub d_ff: usize,
    pub head_dim: usize,
    pub attn_bias: bool,
    pub mlp_bias: bool,
}

#[derive(Deserialize)]
struct RawConfig {
    patch_size: usize,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    layer_group_size: usize,
    num_variate_layers_per_group: usize,
    variate_layer_first: bool,
    #[serde(default)]
    dropout_p: f64,
    #[serde(default = "default_norm_eps")]
    norm_eps: f32,
    #[serde(default)]
    attn_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default = "one")]
    num_output_patches: usize,
    #[serde(default = "yes")]
    pre_norm: bool,
    d_ff: Option<usize>,
    qk_dim: Option<usize>,
    v_dim: Option<usize>,
    num_groups: Option<usize>,
    #[serde(default = "yes")]
    qk_norm: bool,
    #[serde(default)]
    norm_include_weight: bool,
    #[serde(default)]
    per_dim_scale: bool,
    #[serde(default)]
    use_xpos: bool,
}

fn default_norm_eps() -> f32 {
    5e-5
}
fn one() -> usize {
    1
}
fn yes() -> bool {
    true
}

impl Toto2Config {
    /// Parses the published `config.json` and refuses configurations this implementation does not
    /// compute exactly.
    pub fn from_config_json(text: &str) -> Result<Self, InferenceError> {
        let raw: RawConfig = serde_json::from_str(text)
            .map_err(|e| InferenceError::InvalidInput(format!("config.json: {e}")))?;
        let unsupported =
            |why: &str| Err(InferenceError::UnsupportedModel(format!("Toto 2.0: {why}")));
        if raw.dropout_p != 0.0 {
            return unsupported("non-zero dropout_p");
        }
        if !raw.pre_norm || raw.qk_norm || raw.norm_include_weight {
            return unsupported(
                "only pre-norm layers without query/key norms or norm weights are implemented",
            );
        }
        if !raw.per_dim_scale || !raw.use_xpos {
            return unsupported("only per_dim_scale with xPos rotary positions is implemented");
        }
        if raw.num_output_patches != 1 {
            return unsupported("only one output patch per position is implemented");
        }
        if raw.num_heads == 0 || raw.d_model == 0 || !raw.d_model.is_multiple_of(raw.num_heads) {
            return unsupported("d_model must be a non-zero multiple of num_heads");
        }
        let head_dim = raw.qk_dim.unwrap_or(raw.d_model / raw.num_heads);
        if raw.v_dim.unwrap_or(head_dim) != head_dim {
            return unsupported("only equal query/key and value widths are implemented");
        }
        if raw.num_groups.unwrap_or(raw.num_heads) != raw.num_heads {
            return unsupported("grouped-query attention is not implemented");
        }
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return unsupported("head width must be a non-zero multiple of 4");
        }
        if raw.patch_size == 0
            || raw.num_layers == 0
            || raw.layer_group_size == 0
            || !raw.num_layers.is_multiple_of(raw.layer_group_size)
            || raw.num_variate_layers_per_group > raw.layer_group_size
        {
            return unsupported("inconsistent layer grouping");
        }
        Ok(Self {
            patch_size: raw.patch_size,
            d_model: raw.d_model,
            num_heads: raw.num_heads,
            num_layers: raw.num_layers,
            layer_group_size: raw.layer_group_size,
            num_variate_layers_per_group: raw.num_variate_layers_per_group,
            variate_layer_first: raw.variate_layer_first,
            norm_eps: raw.norm_eps,
            d_ff: raw
                .d_ff
                .unwrap_or((4 * raw.d_model * 2 / 3).div_ceil(8) * 8),
            head_dim,
            attn_bias: raw.attn_bias,
            mlp_bias: raw.mlp_bias,
        })
    }

    fn is_variate_layer(&self, layer: usize) -> bool {
        let at = layer % self.layer_group_size;
        if self.variate_layer_first {
            at < self.num_variate_layers_per_group
        } else {
            at >= self.layer_group_size - self.num_variate_layers_per_group
        }
    }
}

/// A unit-scaled projection: `y = (x Wᵀ + b) · scale`, with `W` stored `[out, in]`.
#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
    scale: f32,
}

impl Linear {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * self.out_dim];
        matmul_bt(x, &self.weight, &mut y, rows, self.in_dim, self.out_dim);
        for row in y.chunks_exact_mut(self.out_dim) {
            match &self.bias {
                Some(bias) => {
                    for (v, b) in row.iter_mut().zip(bias) {
                        *v = (*v + b) * self.scale;
                    }
                }
                None => {
                    for v in row.iter_mut() {
                        *v *= self.scale;
                    }
                }
            }
        }
        y
    }
}

/// Unit-scaled SiLU: `silu(x)` times the reference's fixed output scale for `mult = 1`.
fn unit_silu(x: f32) -> f32 {
    let alpha = 1.0 / (1.0 + 0.25);
    let lower = 2.0f64;
    let upper = (2.0 / (1.0 - 1.0 / std::f64::consts::PI)).sqrt();
    let scale = (alpha * upper.ln() + (1.0 - alpha) * lower.ln()).exp() as f32;
    x / (1.0 + (-x).exp()) * scale
}

/// `(h + skip) / √2`: the residual MLP's own merge, with `τ = 1`.
#[derive(Debug, Clone)]
struct ResidualMlp {
    linear1: Linear,
    linear2: Linear,
    skip: Linear,
}

impl ResidualMlp {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = self.linear1.apply(x, rows);
        for v in &mut h {
            *v = unit_silu(*v);
        }
        let mut y = self.linear2.apply(&h, rows);
        let s = self.skip.apply(x, rows);
        let k = std::f32::consts::FRAC_1_SQRT_2;
        for (a, b) in y.iter_mut().zip(&s) {
            *a = *a * k + b * k;
        }
        y
    }
}

#[derive(Debug, Clone)]
struct Layer {
    variate: bool,
    in_proj: Linear,
    out_proj: Linear,
    query_scale: Vec<f32>,
    attn_tau: f32,
    fc1: Linear,
    fc2: Linear,
    mlp_tau: f32,
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
                "Toto 2.0: tensor {name} has shape {got:?}, expected {shape:?}"
            )));
        }
        self.used.push(name.to_string());
        Ok(data.to_vec())
    }

    /// `scale_power` 0.5 for hidden projections (`1/√fan_in`), 1.0 for readouts (`1/fan_in`).
    fn linear(
        &mut self,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
        bias: bool,
        scale_power: f32,
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
            scale: (in_dim as f32).powf(-scale_power),
        })
    }

    fn residual_mlp(
        &mut self,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        hidden: usize,
        readout: bool,
    ) -> Result<ResidualMlp, InferenceError> {
        let out_power = if readout { 1.0 } else { 0.5 };
        Ok(ResidualMlp {
            linear1: self.linear(&format!("{prefix}.linear1"), hidden, in_dim, true, 0.5)?,
            linear2: self.linear(
                &format!("{prefix}.linear2"),
                out_dim,
                hidden,
                true,
                out_power,
            )?,
            skip: self.linear(
                &format!("{prefix}.skip_proj"),
                out_dim,
                in_dim,
                true,
                out_power,
            )?,
        })
    }

    fn scalar(&mut self, name: &str) -> Result<f32, InferenceError> {
        let v = self.get(name, &[])?;
        v.first().copied().ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!("Toto 2.0: tensor {name} is empty"))
        })
    }
}

/// One variate of a call: its history, oldest first, with `NaN` marking a missing value.
#[derive(Debug, Clone, Default)]
pub struct Toto2Variate {
    pub context: Vec<f32>,
    /// Variates with equal ids are one multivariate series and attend to each other.
    pub series_id: u64,
}

/// Post-processing knobs of the reference's forecasting wrapper, defaulted to the values its
/// published evaluations use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Toto2Options {
    /// Leading patches whose cumulative observation count is below this take their location and
    /// scale from the first this-many observations instead. 0 disables the fallback.
    pub scaler_fallback_min_obs: usize,
    /// Each quantile is clipped to the observed context range widened by this many anchor scales.
    /// 0 disables the clip.
    pub quantile_cap_k: f32,
    /// Forecast steps decoded per pass, a multiple of the patch size. 0, or a value covering
    /// the horizon, forecasts in one pass. A smaller value decodes block by block, feeding each
    /// block's median back as observed data and reusing the attention keys and values of the
    /// context through a cache.
    pub decode_block_size: usize,
}

impl Default for Toto2Options {
    fn default() -> Self {
        Self {
            scaler_fallback_min_obs: 8,
            quantile_cap_k: 1e4,
            decode_block_size: 0,
        }
    }
}

/// Network states recorded during a call, for comparison against a reference implementation.
/// Every per-position tensor is laid out `[variate, position, d_model]`. With block decoding the
/// states are those of the first pass.
#[derive(Debug, Clone, Default)]
pub struct Toto2Trace {
    /// The scaler's location and scale before any fallback, `[variate, step]` over the padded
    /// context and the forecast region.
    pub loc: Vec<f32>,
    pub scale: Vec<f32>,
    /// Patch embeddings.
    pub embeddings: Vec<f32>,
    /// Each layer's output.
    pub layers: Vec<Vec<f32>>,
    /// The final normalised state.
    pub final_state: Vec<f32>,
}

/// A loaded Toto 2.0 model.
#[derive(Debug, Clone)]
pub struct Toto2 {
    config: Toto2Config,
    patch_proj: ResidualMlp,
    layers: Vec<Layer>,
    head: ResidualMlp,
    ones: Vec<f32>,
}

/// Keys (after the rotary and xPos scaling) and values of every time-attention layer, kept
/// across the passes of block decoding.
struct KvCache {
    /// `[time layer][variate]`, rows of `num_heads * head_dim`.
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
    /// Rows kept from earlier passes.
    len: usize,
    /// Per variate, whether each position of the first pass may be attended to by later passes;
    /// later positions always may.
    valid: Vec<Vec<bool>>,
}

impl KvCache {
    fn new(time_layers: usize, n_var: usize) -> Self {
        Self {
            keys: vec![vec![Vec::new(); n_var]; time_layers],
            values: vec![vec![Vec::new(); n_var]; time_layers],
            len: 0,
            valid: Vec::new(),
        }
    }
}

/// The reference's causal standard scaler for one row: cumulative mean and Welford variance over
/// observed steps in f64, sample correction 1, floor `1e-6`, each patch taking the statistics of
/// its last step.
fn causal_loc_scale(x: &[f32], mask: &[bool], patch: usize) -> (Vec<f32>, Vec<f32>) {
    let mut loc = vec![0.0f32; x.len()];
    let mut scale = vec![0.0f32; x.len()];
    let (mut sum, mut m2, mut n, mut prev) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (t, (&v, &m)) in x.iter().zip(mask).enumerate() {
        let v = f64::from(v);
        let m = if m { 1.0 } else { 0.0 };
        sum += v * m;
        n += m;
        let cur = sum / n.max(1.0);
        m2 += (v - prev) * (v - cur) * m;
        prev = cur;
        let sd = (m2 / (n - 1.0).max(1.0)).sqrt().max(MIN_SCALE);
        if (t + 1) % patch == 0 {
            let start = t + 1 - patch;
            loc[start..=t].fill(cur as f32);
            scale[start..=t].fill(sd as f32);
        }
    }
    (loc, scale)
}

/// Replaces the statistics of leading patches with fewer than `min_obs` cumulative observations
/// by those of the first `min_obs` observations (f32, sample correction 1).
fn backfill_short_patches(
    x: &[f32],
    mask: &[bool],
    loc: &mut [f32],
    scale: &mut [f32],
    patch: usize,
    min_obs: usize,
) {
    if min_obs == 0 {
        return;
    }
    let mut seen = 0usize;
    let mut first = Vec::with_capacity(min_obs);
    for (&v, &m) in x.iter().zip(mask) {
        if m {
            seen += 1;
            if seen <= min_obs {
                first.push(v);
            }
        }
    }
    let n = first.len().max(1) as f32;
    let donor_loc = first.iter().sum::<f32>() / n;
    let ss: f32 = first
        .iter()
        .map(|v| (v - donor_loc) * (v - donor_loc))
        .sum();
    let donor_scale = (ss / (n - 1.0).max(1.0)).sqrt().max(MIN_SCALE as f32);
    let mut cumulative = 0usize;
    for (p, chunk) in mask.chunks(patch).enumerate() {
        cumulative += chunk.iter().filter(|m| **m).count();
        if cumulative < min_obs {
            loc[p * patch..(p + 1) * patch].fill(donor_loc);
            scale[p * patch..(p + 1) * patch].fill(donor_scale);
        }
    }
}

/// Softmax over the allowed entries of `row`; disallowed entries become 0. At least one entry is
/// always allowed (a position always sees itself).
fn masked_softmax(row: &mut [f32], allowed: impl Fn(usize) -> bool) {
    let max = row
        .iter()
        .enumerate()
        .filter(|(j, _)| allowed(*j))
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (j, v) in row.iter_mut().enumerate() {
        *v = if allowed(j) { (*v - max).exp() } else { 0.0 };
        sum += *v;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

impl Toto2 {
    /// Loads `config.json` and `model.safetensors` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, InferenceError> {
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let config = Toto2Config::from_config_json(&text)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Self::from_safetensors(config, &file)
    }

    /// Builds the model from already-parsed tensors. Every tensor the configuration needs must be
    /// present with its exact shape, and no tensor may be left over.
    pub fn from_safetensors(
        config: Toto2Config,
        file: &SafetensorsFile,
    ) -> Result<Self, InferenceError> {
        let mut t = Tensors {
            file,
            used: Vec::new(),
        };
        let (d, p, hd, h) = (
            config.d_model,
            config.patch_size,
            config.head_dim,
            config.num_heads,
        );
        let patch_proj = t.residual_mlp("patch_proj", 2 * p, d, 4 * d, false)?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let b = format!("transformer.layers.{i}");
            let raw_scale = t.get(&format!("{b}.attn._pds.per_dim_scale"), &[hd])?;
            layers.push(Layer {
                variate: config.is_variate_layer(i),
                in_proj: t.linear(
                    &format!("{b}.attn.in_proj"),
                    3 * h * hd,
                    d,
                    config.attn_bias,
                    0.5,
                )?,
                out_proj: t.linear(
                    &format!("{b}.attn.out_proj"),
                    d,
                    h * hd,
                    config.attn_bias,
                    0.5,
                )?,
                query_scale: raw_scale
                    .iter()
                    .map(|v| (v.exp().ln_1p()) / std::f32::consts::LN_2)
                    .collect(),
                attn_tau: t.scalar(&format!("{b}.attn_tau"))?,
                fc1: t.linear(
                    &format!("{b}.ffn.fc1"),
                    2 * config.d_ff,
                    d,
                    config.mlp_bias,
                    0.5,
                )?,
                fc2: t.linear(
                    &format!("{b}.ffn.fc2"),
                    d,
                    config.d_ff,
                    config.mlp_bias,
                    0.5,
                )?,
                mlp_tau: t.scalar(&format!("{b}.mlp_tau"))?,
            });
        }
        let head = t.residual_mlp(
            "output_head.param_projection.proj",
            d,
            p * QUANTILES.len(),
            4 * d,
            true,
        )?;

        let mut leftover: Vec<&str> = file
            .tensor_names()
            .into_iter()
            .filter(|name| !t.used.iter().any(|u| u == name))
            .collect();
        if !leftover.is_empty() {
            leftover.sort_unstable();
            return Err(InferenceError::InvalidSafetensors(format!(
                "Toto 2.0: tensors not used by this configuration: {}",
                leftover.join(", ")
            )));
        }
        Ok(Self {
            ones: vec![1.0; d],
            config,
            patch_proj,
            layers,
            head,
        })
    }

    pub fn config(&self) -> &Toto2Config {
        &self.config
    }

    /// The quantile levels of every forecast, in output order.
    pub fn quantile_levels(&self) -> &'static [f32] {
        &QUANTILES
    }

    /// Forecasts `horizon` steps for every variate with the reference's default options.
    ///
    /// All contexts must have the same length; shorter histories are padded on the left with
    /// `NaN`. One forecast is returned per variate, in input order.
    pub fn forecast(
        &self,
        variates: &[Toto2Variate],
        horizon: usize,
    ) -> Result<Vec<QuantileForecast>, InferenceError> {
        self.run(variates, horizon, Toto2Options::default(), None)
    }

    /// [`forecast`](Self::forecast) with explicit options.
    pub fn forecast_with_options(
        &self,
        variates: &[Toto2Variate],
        horizon: usize,
        options: Toto2Options,
    ) -> Result<Vec<QuantileForecast>, InferenceError> {
        self.run(variates, horizon, options, None)
    }

    /// [`forecast`](Self::forecast), also returning the network states.
    pub fn forecast_with_trace(
        &self,
        variates: &[Toto2Variate],
        horizon: usize,
    ) -> Result<(Vec<QuantileForecast>, Toto2Trace), InferenceError> {
        let mut trace = Toto2Trace::default();
        let out = self.run(variates, horizon, Toto2Options::default(), Some(&mut trace))?;
        Ok((out, trace))
    }

    fn run(
        &self,
        variates: &[Toto2Variate],
        horizon: usize,
        options: Toto2Options,
        mut trace: Option<&mut Toto2Trace>,
    ) -> Result<Vec<QuantileForecast>, InferenceError> {
        let cfg = &self.config;
        let (d, p) = (cfg.d_model, cfg.patch_size);
        if variates.is_empty() {
            return Ok(Vec::new());
        }
        let raw_len = variates[0].context.len();
        if raw_len == 0 || variates.iter().any(|v| v.context.len() != raw_len) {
            return Err(InferenceError::InvalidInput(
                "Toto 2.0: every context must be non-empty and of equal length".into(),
            ));
        }
        if horizon == 0 {
            return Err(InferenceError::InvalidInput(
                "Toto 2.0: horizon must be at least 1".into(),
            ));
        }
        if !options.decode_block_size.is_multiple_of(p) {
            return Err(InferenceError::InvalidInput(format!(
                "Toto 2.0: decode block size {} is not a multiple of the patch size {p}",
                options.decode_block_size
            )));
        }
        let n_var = variates.len();
        let ctx_patches = raw_len.div_ceil(p);
        let out_patches = horizon.div_ceil(p);
        let seq = ctx_patches + out_patches;
        if seq > MAX_POSITIONS {
            return Err(InferenceError::InvalidInput(format!(
                "Toto 2.0: {ctx_patches} context and {out_patches} forecast patches exceed \
                 {MAX_POSITIONS} positions"
            )));
        }
        let ctx_len = ctx_patches * p;
        let total = seq * p;
        let pad = ctx_len - raw_len;
        let block_patches = match options.decode_block_size {
            0 => out_patches,
            size => (size / p).min(out_patches),
        };

        // Inputs as the reference's wrapper feeds them: left padding and leading gaps are 0,
        // later gaps carry the last observation forward, padding and gaps are unobserved, and the
        // last context patch is then marked observed whatever it holds.
        let mut target = vec![0.0f32; n_var * total];
        let mut mask = vec![false; n_var * total];
        for (v, var) in variates.iter().enumerate() {
            let row = &mut target[v * total..(v + 1) * total];
            let mrow = &mut mask[v * total..(v + 1) * total];
            let mut last = 0.0f32;
            for (i, &x) in var.context.iter().enumerate() {
                if !x.is_nan() {
                    last = x;
                    mrow[pad + i] = true;
                }
                row[pad + i] = last;
            }
            mrow[ctx_len - p..ctx_len].fill(true);
        }

        // Context patches with no observation take group -1: they attend only to each other.
        let mut ids = Vec::with_capacity(n_var);
        for var in variates {
            ids.push(i64::try_from(var.series_id).map_err(|_| {
                InferenceError::InvalidInput("Toto 2.0: series id exceeds i64::MAX".into())
            })?);
        }
        let ctx_observed: Vec<bool> = (0..n_var * ctx_patches)
            .map(|r| {
                let (v, s) = (r / ctx_patches, r % ctx_patches);
                mask[v * total + s * p..v * total + (s + 1) * p]
                    .iter()
                    .any(|m| *m)
            })
            .collect();
        let group_of = |v: usize, s: usize| {
            if s < ctx_patches && !ctx_observed[v * ctx_patches + s] {
                -1
            } else {
                ids[v]
            }
        };

        let nq = QUANTILES.len();
        let median = QUANTILES.iter().position(|q| *q == 0.5).unwrap_or(nq / 2);
        let time_layers = self.layers.iter().filter(|l| !l.variate).count();
        let mut cache = (block_patches < out_patches).then(|| KvCache::new(time_layers, n_var));
        let mut caps: Vec<Option<(f32, f32)>> = Vec::new();
        let mut steps = vec![[0.0f32; 9]; n_var * out_patches * p];
        let mut predicted = 0;
        while predicted < out_patches {
            let block = block_patches.min(out_patches - predicted);

            // The scaler re-runs every block, so predicted medians already filled in count as
            // observations.
            let mut loc = vec![0.0f32; n_var * total];
            let mut scale = vec![0.0f32; n_var * total];
            for v in 0..n_var {
                let r = v * total..(v + 1) * total;
                let (l, s) = causal_loc_scale(&target[r.clone()], &mask[r.clone()], p);
                loc[r.clone()].copy_from_slice(&l);
                scale[r].copy_from_slice(&s);
            }
            if predicted == 0
                && let Some(t) = trace.as_deref_mut()
            {
                t.loc = loc.clone();
                t.scale = scale.clone();
            }
            for v in 0..n_var {
                let r = v * total..(v + 1) * total;
                backfill_short_patches(
                    &target[r.clone()],
                    &mask[r.clone()],
                    &mut loc[r.clone()],
                    &mut scale[r],
                    p,
                    options.scaler_fallback_min_obs,
                );
            }
            if predicted == 0 {
                caps = (0..n_var)
                    .map(|v| {
                        (options.quantile_cap_k > 0.0).then(|| {
                            let row = v * total;
                            let observed = (0..ctx_len)
                                .filter(|&i| mask[row + i])
                                .map(|i| target[row + i]);
                            let hi = observed.clone().fold(f32::NEG_INFINITY, f32::max);
                            let lo = observed.fold(f32::INFINITY, f32::min);
                            let anchor = options.quantile_cap_k * scale[row + ctx_len - 1];
                            let hi = if hi == f32::NEG_INFINITY { 0.0 } else { hi };
                            let lo = if lo == f32::INFINITY { 0.0 } else { lo };
                            (lo - anchor, hi + anchor)
                        })
                    })
                    .collect();
            }

            // Patch embeddings: [asinh of the scaled values, 1 where unobserved] per patch.
            let embed = |patches: std::ops::Range<usize>| {
                let n = patches.len();
                let mut inputs = vec![0.0f32; n_var * n * 2 * p];
                for v in 0..n_var {
                    for (j, s) in patches.clone().enumerate() {
                        let dst = &mut inputs[(v * n + j) * 2 * p..(v * n + j + 1) * 2 * p];
                        for i in 0..p {
                            let at = v * total + s * p + i;
                            if mask[at] {
                                dst[i] = ((target[at] - loc[at]) / scale[at]).asinh();
                            } else {
                                dst[p + i] = 1.0;
                            }
                        }
                    }
                }
                self.patch_proj.apply(&inputs, n_var * n)
            };
            // The first call carries the context and the first block. Later calls carry the
            // previous block's medians (now observed) followed by the new block; the context is
            // read back from the key/value cache.
            let first = if predicted == 0 {
                0
            } else {
                ctx_patches + predicted - block_patches
            };
            let len = ctx_patches + predicted + block - first;
            let x = embed(first..first + len);
            let positions: Vec<usize> = (first..first + len).collect();
            // The reference takes the groups of the first `len` patches here, not of the
            // patches in the call; the two differ only after the first call, where the time mask
            // ignores groups and only the variate mask reads them.
            let mut groups = Vec::with_capacity(n_var * len);
            for v in 0..n_var {
                groups.extend((0..len).map(|s| group_of(v, s)));
            }
            let first_call = if predicted == 0 {
                trace.as_deref_mut()
            } else {
                None
            };
            let x = self.transformer(
                x,
                n_var,
                &positions,
                &groups,
                cache.as_mut(),
                block,
                first_call,
            );

            // The output at position i predicts patch i + 1.
            let mut picked = Vec::with_capacity(n_var * block * d);
            for v in 0..n_var {
                let from = v * len + len - block - 1;
                picked.extend_from_slice(&x[from * d..(from + block) * d]);
            }
            let head = self.head.apply(&picked, n_var * block);
            for v in 0..n_var {
                let row = v * total;
                for k in 0..block {
                    let patch = predicted + k;
                    let base = (v * block + k) * p * nq;
                    let mut real = vec![[0.0f32; 9]; p];
                    for (i, r) in real.iter_mut().enumerate() {
                        let at = row + ctx_len + patch * p + i;
                        for (q, slot) in r.iter_mut().enumerate() {
                            *slot = head[base + i * nq + q].sinh() * scale[at] + loc[at];
                        }
                    }
                    for q in 0..nq {
                        let finite = || real.iter().map(|r| r[q]).filter(|v| v.is_finite());
                        let max = finite().fold(f32::NEG_INFINITY, f32::max);
                        let min = finite().fold(f32::INFINITY, f32::min);
                        for r in real.iter_mut() {
                            if r[q] == f32::INFINITY {
                                r[q] = max;
                            } else if r[q] == f32::NEG_INFINITY {
                                r[q] = min;
                            }
                        }
                    }
                    for (i, r) in real.iter_mut().enumerate() {
                        if let Some((lo, hi)) = caps[v] {
                            for v in r.iter_mut() {
                                *v = v.clamp(lo, hi);
                            }
                        }
                        r.sort_by(f32::total_cmp);
                        steps[(v * out_patches + patch) * p + i] = *r;
                    }
                }
            }
            predicted += block;

            // Feed the block's medians back as observations for the next block.
            if predicted < out_patches {
                for v in 0..n_var {
                    for t in (predicted - block) * p..predicted * p {
                        let at = v * total + ctx_len + t;
                        target[at] = steps[v * out_patches * p + t][median];
                        mask[at] = true;
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(n_var);
        for v in 0..n_var {
            let rows = &steps[v * out_patches * p..(v + 1) * out_patches * p];
            let mut values = vec![0.0f32; nq * horizon];
            for (t, s) in rows.iter().take(horizon).enumerate() {
                for q in 0..nq {
                    values[q * horizon + t] = s[q];
                }
            }
            out.push(QuantileForecast { horizon, values });
        }
        Ok(out)
    }

    /// One pass of the decoder over `positions.len()` patches per variate. With a cache, keys and
    /// values are appended to it and the last `block` positions are dropped again afterwards, so
    /// it keeps only the patches that later calls re-read.
    #[allow(clippy::too_many_arguments)]
    fn transformer(
        &self,
        mut x: Vec<f32>,
        n_var: usize,
        positions: &[usize],
        groups: &[i64],
        mut cache: Option<&mut KvCache>,
        block: usize,
        mut trace: Option<&mut Toto2Trace>,
    ) -> Vec<f32> {
        let (d, eps) = (self.config.d_model, self.config.norm_eps);
        let len = positions.len();
        if let Some(t) = trace.as_deref_mut() {
            t.embeddings = x.clone();
        }
        if let Some(c) = cache.as_deref_mut()
            && c.len == 0
        {
            c.valid = groups
                .chunks(len)
                .map(|g| g.iter().map(|g| *g != -1).collect())
                .collect();
        }
        let tables = self.rotary_tables(positions);
        let mut time_layer = 0;
        for layer in &self.layers {
            let mut h = x.clone();
            rms_norm(&mut h, &self.ones, d, eps);
            let a = if layer.variate {
                self.variate_attention(layer, &h, groups, n_var, len)
            } else {
                let a = self.time_attention(
                    layer,
                    &h,
                    groups,
                    n_var,
                    len,
                    (&tables.0, &tables.1, &tables.2, &tables.3),
                    cache.as_deref_mut().map(|c| (c, time_layer)),
                    block,
                );
                time_layer += 1;
                a
            };
            merge(&mut x, &a, layer.attn_tau);
            let mut h = x.clone();
            rms_norm(&mut h, &self.ones, d, eps);
            let f = self.feed_forward(layer, &h, n_var * len);
            merge(&mut x, &f, layer.mlp_tau);
            if let Some(t) = trace.as_deref_mut() {
                t.layers.push(x.clone());
            }
        }
        rms_norm(&mut x, &self.ones, d, eps);
        if let Some(t) = trace {
            t.final_state = x.clone();
        }
        if let Some(c) = cache {
            c.len += len - block;
        }
        x
    }

    /// Interleaved rotary tables over the first half of each head and the xPos scales for queries
    /// (`ζ^((pos - c)/256)`) and keys (the reciprocal), per call position and pair. The centre is
    /// `c = ⌊(max pos + 1)/2⌋` of this call, so keys cached by an earlier call keep that call's
    /// centre, as in the reference.
    fn rotary_tables(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let width = self.config.head_dim / 2;
        let pairs = width / 2;
        let len = positions.len();
        let center = positions.iter().copied().max().unwrap_or(0).div_ceil(2) as f32;
        let mut cos = vec![0.0f32; len * pairs];
        let mut sin = vec![0.0f32; len * pairs];
        let mut qx = vec![0.0f32; len * pairs];
        let mut kx = vec![0.0f32; len * pairs];
        for (r, &pos) in positions.iter().enumerate() {
            let power = (pos as f32 - center) / XPOS_SCALE_BASE;
            for i in 0..pairs {
                let theta = 1.0 / ROPE_BASE.powf((2 * i) as f32 / width as f32);
                let angle = pos as f32 * theta;
                cos[r * pairs + i] = angle.cos();
                sin[r * pairs + i] = angle.sin();
                let zeta = ((2 * i) as f32 + 0.4 * width as f32) / (1.4 * width as f32);
                let s = zeta.powf(power);
                qx[r * pairs + i] = s;
                kx[r * pairs + i] = s.powf(-1.0);
            }
        }
        (cos, sin, qx, kx)
    }

    /// Splits the fused projection into per-head query, key and value rows and applies the learned
    /// per-dimension query scale.
    fn project(&self, layer: &Layer, h: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let width = self.config.num_heads * self.config.head_dim;
        let hd = self.config.head_dim;
        let qkv = layer.in_proj.apply(h, rows);
        let mut q = Vec::with_capacity(rows * width);
        let mut k = Vec::with_capacity(rows * width);
        let mut v = Vec::with_capacity(rows * width);
        for row in qkv.chunks_exact(3 * width) {
            for head in row[..width].chunks_exact(hd) {
                q.extend(head.iter().zip(&layer.query_scale).map(|(a, s)| a * s));
            }
            k.extend_from_slice(&row[width..2 * width]);
            v.extend_from_slice(&row[2 * width..]);
        }
        (q, k, v)
    }

    /// Causal attention along time within each variate. Without a cache, or on the call that
    /// fills it, a patch attends only to earlier patches of its own group. On later calls the
    /// queries see every cached patch the first call marked valid plus the call's own earlier
    /// patches, regardless of group.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn time_attention(
        &self,
        layer: &Layer,
        h: &[f32],
        groups: &[i64],
        n_var: usize,
        len: usize,
        tables: (&[f32], &[f32], &[f32], &[f32]),
        mut cache: Option<(&mut KvCache, usize)>,
        block: usize,
    ) -> Vec<f32> {
        let (cos, sin, q_xpos, k_xpos) = tables;
        let (heads, hd) = (self.config.num_heads, self.config.head_dim);
        let width = heads * hd;
        let pairs = hd / 4;
        let rows = n_var * len;
        let (mut q, mut k, v) = self.project(layer, h, rows);
        for r in 0..rows {
            let pos = r % len;
            for head in 0..heads {
                let off = r * width + head * hd;
                for i in 0..pairs {
                    let t = pos * pairs + i;
                    for (buf, xpos) in [(&mut q, q_xpos), (&mut k, k_xpos)] {
                        let (a, b) = (buf[off + 2 * i], buf[off + 2 * i + 1]);
                        buf[off + 2 * i] = (cos[t] * a - sin[t] * b) * xpos[t];
                        buf[off + 2 * i + 1] = (cos[t] * b + sin[t] * a) * xpos[t];
                    }
                }
            }
        }
        let prior = cache.as_ref().map_or(0, |(c, _)| c.len);
        if let Some((c, li)) = cache.as_mut() {
            for var in 0..n_var {
                let r = var * len * width..(var + 1) * len * width;
                c.keys[*li][var].extend_from_slice(&k[r.clone()]);
                c.values[*li][var].extend_from_slice(&v[r]);
            }
        }
        let inv = 1.0 / hd as f32;
        let mut out = vec![0.0f32; rows * width];
        let mut scores = vec![0.0f32; prior + len];
        for var in 0..n_var {
            let (kk, vv, valid) = match cache.as_ref() {
                Some((c, li)) => (
                    &c.keys[*li][var][..],
                    &c.values[*li][var][..],
                    Some(&c.valid[var][..]),
                ),
                None => (
                    &k[var * len * width..(var + 1) * len * width],
                    &v[var * len * width..(var + 1) * len * width],
                    None,
                ),
            };
            let g = &groups[var * len..(var + 1) * len];
            for head in 0..heads {
                for i in 0..len {
                    let qi = &q[(var * len + i) * width + head * hd..][..hd];
                    let row = &mut scores[..prior + i + 1];
                    for (j, s) in row.iter_mut().enumerate() {
                        let kj = &kk[j * width + head * hd..][..hd];
                        *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * inv;
                    }
                    if prior == 0 {
                        masked_softmax(row, |j| g[j] == g[i]);
                    } else {
                        masked_softmax(row, |j| {
                            valid.is_none_or(|valid| valid.get(j).copied().unwrap_or(true))
                        });
                    }
                    let o = &mut out[(var * len + i) * width + head * hd..][..hd];
                    for (j, w) in row.iter().enumerate() {
                        if *w != 0.0 {
                            let vj = &vv[j * width + head * hd..][..hd];
                            for (a, b) in o.iter_mut().zip(vj) {
                                *a += w * b;
                            }
                        }
                    }
                }
            }
        }
        if let Some((c, li)) = cache.as_mut() {
            let keep = (prior + len - block) * width;
            for var in 0..n_var {
                c.keys[*li][var].truncate(keep);
                c.values[*li][var].truncate(keep);
            }
        }
        layer.out_proj.apply(&out, rows)
    }

    fn variate_attention(
        &self,
        layer: &Layer,
        h: &[f32],
        groups: &[i64],
        n_var: usize,
        seq: usize,
    ) -> Vec<f32> {
        let (heads, hd) = (self.config.num_heads, self.config.head_dim);
        let width = heads * hd;
        let rows = n_var * seq;
        let (q, k, v) = self.project(layer, h, rows);
        let inv = 1.0 / hd as f32;
        let mut out = vec![0.0f32; rows * width];
        let mut scores = vec![0.0f32; n_var];
        for s in 0..seq {
            for head in 0..heads {
                for a in 0..n_var {
                    let qa = &q[(a * seq + s) * width + head * hd..][..hd];
                    for (b, sc) in scores.iter_mut().enumerate() {
                        let kb = &k[(b * seq + s) * width + head * hd..][..hd];
                        *sc = qa.iter().zip(kb).map(|(x, y)| x * y).sum::<f32>() * inv;
                    }
                    let ga = groups[a * seq + s];
                    masked_softmax(&mut scores, |b| groups[b * seq + s] == ga);
                    let o = &mut out[(a * seq + s) * width + head * hd..][..hd];
                    for (b, w) in scores.iter().enumerate() {
                        if *w != 0.0 {
                            let vb = &v[(b * seq + s) * width + head * hd..][..hd];
                            for (x, y) in o.iter_mut().zip(vb) {
                                *x += w * y;
                            }
                        }
                    }
                }
            }
        }
        layer.out_proj.apply(&out, rows)
    }

    /// `fc2(a · silu(b))` where `fc1(x) = [a, b]`.
    fn feed_forward(&self, layer: &Layer, h: &[f32], rows: usize) -> Vec<f32> {
        let ff = self.config.d_ff;
        let up = layer.fc1.apply(h, rows);
        let mut gated = Vec::with_capacity(rows * ff);
        for row in up.chunks_exact(2 * ff) {
            let (gate, x) = row.split_at(ff);
            gated.extend(
                gate.iter()
                    .zip(x)
                    .map(|(g, x)| g * (x / (1.0 + (-x).exp()))),
            );
        }
        layer.fc2.apply(&gated, rows)
    }
}

/// `x ← f · τ/√(1+τ²) + x / √(1+τ²)`: the reference's τ-weighted residual merge.
fn merge(x: &mut [f32], f: &[f32], tau: f32) {
    let denom = (1.0 + tau * tau).sqrt();
    let (a, b) = (tau / denom, 1.0 / denom);
    for (x, f) in x.iter_mut().zip(f) {
        *x = f * a + *x * b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_22M: &str = r#"{"attn_bias": true, "d_ff": 1368, "d_model": 512, "dropout_p": 0.0,
        "heads_per_group": 1, "layer_group_size": 6, "mlp_bias": false, "norm_eps": 0.0001,
        "norm_include_weight": false, "num_groups": 8, "num_heads": 8, "num_layers": 6,
        "num_output_patches": 1, "num_variate_layers_per_group": 1, "patch_size": 32,
        "per_dim_scale": true, "pre_norm": true, "qk_dim": 64, "qk_norm": false,
        "qk_norm_include_weight": false, "residual_attn_ratio": 5.136215466577748,
        "residual_mult": 0.75, "use_xpos": true, "v_dim": 64, "variate_layer_first": false}"#;

    #[test]
    fn config_parses_and_places_the_variate_layer_last() {
        let c = Toto2Config::from_config_json(CONFIG_22M).unwrap();
        assert_eq!(
            (c.d_model, c.num_heads, c.head_dim, c.d_ff),
            (512, 8, 64, 1368)
        );
        let variate: Vec<usize> = (0..c.num_layers)
            .filter(|&i| c.is_variate_layer(i))
            .collect();
        assert_eq!(variate, vec![5]);
    }

    #[test]
    fn config_refuses_what_is_not_implemented() {
        for (from, to) in [
            ("\"qk_norm\": false", "\"qk_norm\": true"),
            ("\"num_groups\": 8", "\"num_groups\": 4"),
            ("\"use_xpos\": true", "\"use_xpos\": false"),
            ("\"num_output_patches\": 1", "\"num_output_patches\": 2"),
            (
                "\"norm_include_weight\": false",
                "\"norm_include_weight\": true",
            ),
        ] {
            let text = CONFIG_22M.replacen(from, to, 1);
            assert_ne!(text, CONFIG_22M, "fixture edit {to} did not apply");
            assert!(
                matches!(
                    Toto2Config::from_config_json(&text),
                    Err(InferenceError::UnsupportedModel(_))
                ),
                "{to} was accepted"
            );
        }
    }

    #[test]
    fn the_scaler_is_causal_per_patch_and_floors_a_flat_series() {
        let x = [1.0, 3.0, 5.0, 7.0, 2.0, 2.0];
        let m = [true, true, true, true, false, true];
        let (loc, scale) = causal_loc_scale(&x, &m, 2);
        assert_eq!(loc, vec![2.0, 2.0, 4.0, 4.0, 3.6, 3.6]);
        assert!((scale[0] - 2f32.sqrt()).abs() < 1e-6);
        assert!((scale[2] - (20.0f32 / 3.0).sqrt()).abs() < 1e-5);
        let (_, flat) = causal_loc_scale(&[5.0; 4], &[true; 4], 2);
        assert_eq!(flat, vec![1e-6; 4]);
    }

    #[test]
    fn short_leading_patches_take_the_first_observations_statistics() {
        let x = [0.0, 0.0, 1.0, 3.0, 5.0, 7.0];
        let m = [false, false, true, true, true, true];
        let (mut loc, mut scale) = causal_loc_scale(&x, &m, 2);
        backfill_short_patches(&x, &m, &mut loc, &mut scale, 2, 3);
        assert_eq!(&loc[..4], &[3.0; 4]);
        assert_eq!(&scale[..4], &[2.0; 4]);
        assert_eq!(loc[4], 4.0);
    }

    #[test]
    fn unit_silu_matches_the_reference_scale() {
        let s = unit_silu(1.0) / (1.0 / (1.0 + (-1.0f32).exp()));
        assert!((s - 1.766_78).abs() < 1e-4, "{s}");
    }
}
