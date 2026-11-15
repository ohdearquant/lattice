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

use lattice_compute::forward::cpu::{matmul_bt, rms_norm};
use lattice_inference_core::InferenceError;
use lattice_weights::SafetensorsFile;

const SCALE_FLOOR: f32 = 1e-5;
const REG_TOKEN_ID: usize = 1;
const HORIZON_OVERFLOW_MESSAGE: &str =
    "Chronos-2: max_output_patches * patch_size exceeds usize::MAX";

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
        config.checked_max_horizon()?;
        Ok(config)
    }

    /// Steps of horizon one call covers, or an error when the product is unrepresentable.
    pub fn checked_max_horizon(&self) -> Result<usize, InferenceError> {
        self.max_output_patches
            .checked_mul(self.patch_size)
            .ok_or_else(|| InferenceError::InvalidInput(HORIZON_OVERFLOW_MESSAGE.into()))
    }

    /// Steps of horizon one call covers.
    /// Configurations that did not pass the parser can saturate to `usize::MAX` here.
    #[deprecated(note = "use checked_max_horizon")]
    pub fn max_horizon(&self) -> usize {
        self.max_output_patches.saturating_mul(self.patch_size)
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
    request_memory_budget: Option<usize>,
    reg_embedding: Option<Vec<f32>>,
    input_embedding: ResidualBlock,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    output_embedding: ResidualBlock,
    inv_freq: Vec<f32>,
}

#[derive(Debug)]
struct RequestExtents {
    horizon_steps: usize,
    context_patch_count: usize,
    padded_context_steps: usize,
    sequence_position_count: usize,
    context_row_count: usize,
    future_row_count: usize,
    encoder_row_count: usize,
    key_mask_elements: usize,
    context_input_elements: usize,
    future_input_elements: usize,
    encoder_hidden_elements: usize,
    future_embedding_elements: usize,
    forecast_value_elements: usize,
}

fn request_product(name: &str, factors: &[usize]) -> Result<usize, InferenceError> {
    factors
        .iter()
        .try_fold(1usize, |n, &factor| n.checked_mul(factor))
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!("Chronos-2: request {name} exceeds usize::MAX"))
        })
}

fn request_buffer_elements<T>(name: &str, factors: &[usize]) -> Result<usize, InferenceError> {
    let elements = request_product(name, factors)?;
    if elements
        .checked_mul(std::mem::size_of::<T>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(InferenceError::InvalidInput(format!(
            "Chronos-2: request {name} exceeds isize::MAX bytes"
        )));
    }
    Ok(elements)
}

fn check_request_buffer<T>(name: &str, factors: &[usize]) -> Result<(), InferenceError> {
    request_buffer_elements::<T>(name, factors).map(|_| ())
}

/// Checks `usize` element-product overflow and each buffer's `isize::MAX` byte extent before `run` allocates.
///
/// Keep this manual inventory in sync with `run`, `Linear::apply`, `ResidualBlock::apply`,
/// `time_attention`, `group_attention`, `feed_forward`, `rope_tables` and `location_scale`.
/// Allocation changes also require updating the coexisting-storage totals in
/// `request_memory_peak_bytes`.
/// A checked size can cover several same-sized buffers, including encoder copies and
/// `forward_with_trace` snapshots. The output-prediction check also covers the total
/// forecast-value element count across the batch. These checks do not impose a configured
/// aggregate memory budget for workspaces, trace snapshots, output predictions and
/// returned forecast values that coexist.
fn checked_request_extents(
    cfg: &Chronos2Config,
    bsz: usize,
    len: usize,
    num_output_patches: usize,
    has_blocks: bool,
) -> Result<RequestExtents, InferenceError> {
    let p = cfg.patch_size;
    if p == 0 {
        return Err(InferenceError::InvalidInput(
            "Chronos-2: request patch size must be non-zero".into(),
        ));
    }
    let horizon_steps = request_product("horizon", &[num_output_patches, p])?;
    let context_patch_count = len.div_ceil(p);
    let padded_context_steps = request_product("padded context", &[context_patch_count, p])?;
    let sequence_position_count = context_patch_count
        .checked_add(usize::from(cfg.use_reg_token))
        .and_then(|n| n.checked_add(num_output_patches))
        .ok_or_else(|| {
            InferenceError::InvalidInput(
                "Chronos-2: request sequence length exceeds usize::MAX".into(),
            )
        })?;
    let context_row_count = request_product("context rows", &[bsz, context_patch_count])?;
    let future_row_count = request_product("future rows", &[bsz, num_output_patches])?;
    let context_input_elements =
        request_buffer_elements::<f32>("context patch inputs", &[context_row_count, 3, p])?;
    let future_input_elements =
        request_buffer_elements::<f32>("future patch inputs", &[future_row_count, 3, p])?;
    let key_mask_elements =
        request_buffer_elements::<f32>("key mask", &[bsz, sequence_position_count])?;
    let encoder_row_count = key_mask_elements;
    check_request_buffer::<f32>("context embedding hidden", &[context_row_count, cfg.d_ff])?;
    check_request_buffer::<f32>("context embeddings", &[context_row_count, cfg.d_model])?;
    check_request_buffer::<f32>("future embedding hidden", &[future_row_count, cfg.d_ff])?;
    let future_embedding_elements =
        request_buffer_elements::<f32>("future embeddings", &[future_row_count, cfg.d_model])?;
    let encoder_hidden_elements =
        request_buffer_elements::<f32>("encoder hidden", &[encoder_row_count, cfg.d_model])?;
    check_request_buffer::<f32>("rotary tables", &[sequence_position_count, cfg.d_kv])?;
    if has_blocks {
        let inner = request_product("attention width", &[cfg.num_heads, cfg.d_kv])?;
        check_request_buffer::<f32>("attention projections", &[encoder_row_count, inner])?;
        check_request_buffer::<f32>(
            "time attention scores",
            &[sequence_position_count, sequence_position_count],
        )?;
        check_request_buffer::<f32>("feed-forward hidden", &[encoder_row_count, cfg.d_ff])?;
    }
    let quantile_patch = request_product("quantile patch", &[cfg.quantiles.len(), p])?;
    check_request_buffer::<f32>(
        "output patch predictions",
        &[future_row_count, quantile_patch],
    )?;
    let forecast_value_elements =
        request_buffer_elements::<f32>("forecast values", &[cfg.quantiles.len(), horizon_steps])?;
    check_request_buffer::<(f32, f32)>("location and scale", &[bsz])?;
    check_request_buffer::<u64>("group ids", &[bsz])?;
    check_request_buffer::<QuantileForecast>("forecast batch", &[bsz])?;
    Ok(RequestExtents {
        horizon_steps,
        context_patch_count,
        padded_context_steps,
        sequence_position_count,
        context_row_count,
        future_row_count,
        encoder_row_count,
        key_mask_elements,
        context_input_elements,
        future_input_elements,
        encoder_hidden_elements,
        future_embedding_elements,
        forecast_value_elements,
    })
}

// Owning Vec lifetimes extend through their enclosing phase, independently of last borrow use.
// Keep these logical/requested extents in sync with the allocation inventory above.
fn request_memory_peak_bytes(
    cfg: &Chronos2Config,
    extents: &RequestExtents,
    bsz: usize,
    len: usize,
    blocks: usize,
    traced: bool,
    predict: Option<(usize, usize)>,
) -> Option<usize> {
    fn sum(parts: &[usize]) -> Option<usize> {
        parts.iter().try_fold(0usize, |a, &b| a.checked_add(b))
    }
    let f32_bytes = |n: usize| n.checked_mul(std::mem::size_of::<f32>());
    let context = f32_bytes(extents.context_row_count.checked_mul(cfg.d_model)?)?;
    let future = f32_bytes(extents.future_embedding_elements)?;
    let hidden = f32_bytes(extents.encoder_hidden_elements)?;
    let context_mid = f32_bytes(extents.context_row_count.checked_mul(cfg.d_ff)?)?;
    let future_mid = f32_bytes(extents.future_row_count.checked_mul(cfg.d_ff)?)?;
    let preparation = sum(&[
        f32_bytes(extents.context_input_elements)?,
        f32_bytes(extents.future_input_elements)?,
        f32_bytes(extents.key_mask_elements)?,
        bsz.checked_mul(std::mem::size_of::<(f32, f32)>())?,
    ])?;
    let mut peak = sum(&[preparation, f32_bytes(len)?])?;
    peak = peak.max(sum(&[preparation, context_mid, context.checked_mul(2)?])?);
    peak = peak.max(sum(&[
        preparation,
        context,
        future_mid,
        future.checked_mul(2)?,
    ])?);
    let combined = sum(&[preparation, context, future, hidden])?;
    peak = peak.max(combined);
    let rotary = f32_bytes(extents.sequence_position_count.checked_mul(cfg.d_kv)?)?;
    let base = sum(&[
        combined,
        rotary.checked_mul(2)?,
        bsz.checked_mul(std::mem::size_of::<u64>())?,
    ])?;
    peak = peak.max(sum(&[base, if traced { hidden } else { 0 }])?);
    if blocks > 0 {
        let inner = cfg.num_heads.checked_mul(cfg.d_kv)?;
        let projection = f32_bytes(extents.encoder_row_count.checked_mul(inner)?)?;
        let rotation = sum(&[hidden, projection.checked_mul(3)?, f32_bytes(cfg.d_kv)?])?;
        let time = sum(&[
            hidden.checked_mul(2)?,
            projection.checked_mul(4)?,
            rotary.checked_mul(3)?,
            f32_bytes(
                extents
                    .sequence_position_count
                    .checked_mul(extents.sequence_position_count)?,
            )?,
        ])?;
        let group = sum(&[
            hidden.checked_mul(2)?,
            projection.checked_mul(4)?,
            f32_bytes(bsz)?,
        ])?;
        let feed_forward = sum(&[
            hidden.checked_mul(2)?,
            f32_bytes(extents.encoder_row_count.checked_mul(cfg.d_ff)?)?,
        ])?;
        let retained = if traced {
            sum(&[
                hidden.checked_mul(blocks)?,
                (blocks - 1).checked_mul(std::mem::size_of::<Vec<f32>>())?,
            ])?
        } else {
            0
        };
        peak = peak.max(sum(&[
            base,
            retained,
            rotation.max(time).max(group).max(feed_forward),
        ])?);
    }
    // The final snapshot includes every preceding block snapshot; it also covers their
    // earlier clone phases, which have no additional live helper workspace.
    let trace_final = if traced {
        sum(&[
            hidden.checked_mul(blocks.checked_add(2)?)?,
            blocks.checked_mul(std::mem::size_of::<Vec<f32>>())?,
        ])?
    } else {
        0
    };
    peak = peak.max(sum(&[base, trace_final])?);
    let predictions = f32_bytes(bsz.checked_mul(extents.forecast_value_elements)?)?;
    peak = peak.max(sum(&[
        base,
        trace_final,
        future,
        future_mid,
        predictions.checked_mul(2)?,
    ])?);
    peak = peak.max(sum(&[
        base,
        trace_final,
        future,
        predictions.checked_mul(2)?,
        bsz.checked_mul(std::mem::size_of::<QuantileForecast>())?,
    ])?);
    if let Some((raw_len, horizon)) = predict {
        let copied_context = f32_bytes(raw_len)?;
        let trimming = sum(&[
            copied_context,
            f32_bytes(extents.forecast_value_elements)?,
            f32_bytes(cfg.quantiles.len().checked_mul(horizon)?)?,
            f32_bytes(horizon)?,
        ])?;
        peak = sum(&[peak, copied_context])?.max(trimming);
    }
    Some(peak)
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
        config.checked_max_horizon()?;
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
            request_memory_budget: None,
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

    /// Sets the optional byte budget for accounted peak Chronos-owned request storage.
    ///
    /// Defaults to `None`, which disables aggregate accounting without changing existing
    /// per-buffer checks. Cloning a model preserves this setting. With a budget, equality
    /// is admitted; excess and accounting overflow return [`InferenceError::InvalidInput`].
    ///
    /// The account includes coexisting numeric payloads, heap element descriptors, explicit
    /// requested capacities, retained trace snapshots and `predict`'s copy/trimming buffers.
    /// It excludes vector spare capacity, allocator overhead, backend packing/thread pools,
    /// persistent model state, caller-owned inputs and error formatting. It is not an RSS
    /// limit or an allocation-success guarantee. See ADR-101's Chronos-2 memory amendment.
    pub fn set_request_memory_budget(&mut self, budget: Option<usize>) {
        self.request_memory_budget = budget;
    }

    fn check_request_memory(
        &self,
        extents: &RequestExtents,
        bsz: usize,
        len: usize,
        traced: bool,
        predict: Option<(usize, usize)>,
    ) -> Result<(), InferenceError> {
        let Some(budget) = self.request_memory_budget else {
            return Ok(());
        };
        let total = request_memory_peak_bytes(
            &self.config,
            extents,
            bsz,
            len,
            self.blocks.len(),
            traced,
            predict,
        )
        .ok_or_else(|| {
            InferenceError::InvalidInput(
                "Chronos-2: request memory accounting exceeds usize::MAX".into(),
            )
        })?;
        if total > budget {
            return Err(InferenceError::InvalidInput(format!(
                "Chronos-2: request memory requires {total} bytes, exceeding budget {budget} bytes"
            )));
        }
        Ok(())
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

    /// Forecasts one series on its own for `horizon` steps (at most
    /// [`Chronos2Config::checked_max_horizon`]).
    pub fn predict(
        &self,
        context: &[f32],
        horizon: usize,
    ) -> Result<QuantileForecast, InferenceError> {
        let max_horizon = self.config.checked_max_horizon()?;
        if horizon == 0 || horizon > max_horizon {
            return Err(InferenceError::InvalidInput(format!(
                "Chronos-2: horizon {horizon} is outside 1..={max_horizon}"
            )));
        }
        let patches = horizon.div_ceil(self.config.patch_size);
        if self.request_memory_budget.is_some() {
            if context.is_empty() {
                return Err(InferenceError::InvalidInput(
                    "Chronos-2: contexts must be non-empty and of equal length; pad on the left with NaN"
                        .into(),
                ));
            }
            let len = context.len().min(self.config.context_length);
            let extents =
                checked_request_extents(&self.config, 1, len, patches, !self.blocks.is_empty())?;
            self.check_request_memory(&extents, 1, len, false, Some((context.len(), horizon)))?;
        }
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
        let bsz = batch.len();
        let len = raw_len.min(cfg.context_length);
        let extents =
            checked_request_extents(cfg, bsz, len, num_output_patches, !self.blocks.is_empty())?;
        let horizon = extents.horizon_steps;
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
        self.check_request_memory(&extents, bsz, len, trace.is_some(), None)?;

        let n_ctx = extents.context_patch_count;
        let pad = extents.padded_context_steps - len;
        let reg = usize::from(cfg.use_reg_token);
        let seq = extents.sequence_position_count;

        // Patch inputs [time(p), value(p), mask(p)] for every position that takes one, plus the
        // per-position attention mask and each series' location and scale.
        let mut context_patch_inputs = vec![0.0f32; extents.context_input_elements];
        let mut future_patch_inputs = vec![0.0f32; extents.future_input_elements];
        let mut key_mask = vec![1.0f32; extents.key_mask_elements];
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
                let row =
                    &mut context_patch_inputs[(b * n_ctx + n) * 3 * p..(b * n_ctx + n + 1) * 3 * p];
                let mut any = false;
                for j in 0..p {
                    let pos = n * p + j;
                    row[j] = (pos as f32 - extents.padded_context_steps as f32)
                        / cfg.time_encoding_scale;
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
                let row = &mut future_patch_inputs[(b * num_output_patches + n) * 3 * p
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

        let ctx_embeds = self
            .input_embedding
            .apply(&context_patch_inputs, extents.context_row_count);
        let fut_embeds = self
            .input_embedding
            .apply(&future_patch_inputs, extents.future_row_count);
        let mut h = vec![0.0f32; extents.encoder_hidden_elements];
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
            self.feed_forward(block, &mut h, extents.encoder_row_count);
            if let Some(t) = trace.as_deref_mut() {
                t.blocks.push(h.clone());
            }
        }
        rms_norm(&mut h, &self.final_norm, d, cfg.layer_norm_epsilon);
        if let Some(t) = trace {
            t.final_state = h.clone();
        }

        let nq = cfg.quantiles.len();
        let mut future_encoder_embeddings = vec![0.0f32; extents.future_embedding_elements];
        for b in 0..bsz {
            let src = &h[(b * seq + n_ctx + reg) * d..(b + 1) * seq * d];
            future_encoder_embeddings[b * num_output_patches * d..(b + 1) * num_output_patches * d]
                .copy_from_slice(src);
        }
        let y = self
            .output_embedding
            .apply(&future_encoder_embeddings, extents.future_row_count);
        let mut forecasts = Vec::with_capacity(bsz);
        for (b, &(loc, scale)) in loc_scale.iter().enumerate() {
            let mut values = vec![0.0f32; extents.forecast_value_elements];
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

    fn model_from_parts(
        config: Chronos2Config,
        input_embedding: ResidualBlock,
        final_norm: Vec<f32>,
        output_embedding: ResidualBlock,
        inv_freq: Vec<f32>,
    ) -> Chronos2 {
        Chronos2 {
            config,
            request_memory_budget: None,
            reg_embedding: None,
            input_embedding,
            blocks: Vec::new(),
            final_norm,
            output_embedding,
            inv_freq,
        }
    }

    #[test]
    fn config_parses_and_defaults_the_time_scale_to_the_context_length() {
        let c = Chronos2Config::from_config_json(CONFIG).unwrap();
        assert_eq!((c.d_model, c.d_kv, c.num_heads, c.patch_size), (8, 4, 2, 4));
        assert_eq!(c.time_encoding_scale, 64.0);
        assert_eq!(c.quantiles, vec![0.1, 0.5, 0.9]);
        assert_eq!(c.checked_max_horizon().unwrap(), 12);
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
    fn config_refuses_an_unrepresentable_horizon() {
        let mut value: Value = serde_json::from_str(CONFIG).unwrap();
        value["chronos_config"]["max_output_patches"] = (usize::MAX / 4 + 1).into();
        let result = Chronos2Config::from_config_json(&value.to_string());
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(ref message))
                if message == "Chronos-2: max_output_patches * patch_size exceeds usize::MAX"),
            "Chronos-2 must refuse an unrepresentable horizon: {result:?}"
        );
    }

    #[test]
    fn config_accepts_the_largest_representable_patch_count() {
        let mut value: Value = serde_json::from_str(CONFIG).unwrap();
        value["chronos_config"]["max_output_patches"] = (usize::MAX / 4).into();
        let config = Chronos2Config::from_config_json(&value.to_string())
            .expect("representable horizons must not acquire an arbitrary cap");
        assert_eq!(config.max_output_patches, usize::MAX / 4);
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.checked_max_horizon().unwrap(), usize::MAX / 4 * 4);
    }

    #[test]
    fn checked_horizon_refuses_direct_overflowing_config() {
        let mut config = Chronos2Config::from_config_json(CONFIG).unwrap();
        config.max_output_patches = usize::MAX / config.patch_size + 1;
        assert!(
            matches!(config.checked_max_horizon(), Err(InferenceError::InvalidInput(message))
                if message == HORIZON_OVERFLOW_MESSAGE),
            "direct configurations must report an unrepresentable horizon"
        );
    }

    #[test]
    fn model_constructor_checks_horizon_before_reading_tensors() {
        let mut bytes = 8u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}      ");
        let file = SafetensorsFile::from_bytes(bytes).unwrap();
        let mut config = Chronos2Config::from_config_json(CONFIG).unwrap();
        config.max_output_patches = usize::MAX / config.patch_size + 1;
        assert!(
            matches!(Chronos2::from_safetensors(config, &file),
                Err(InferenceError::InvalidInput(message))
                if message == HORIZON_OVERFLOW_MESSAGE),
            "direct overflowing configs must be refused before missing tensors are read"
        );
    }

    #[test]
    fn predict_checks_a_direct_config_before_computation() {
        let linear = Linear {
            weight: Vec::new(),
            bias: None,
            in_dim: 0,
            out_dim: 0,
        };
        let embedding = ResidualBlock {
            hidden: linear.clone(),
            output: linear.clone(),
            residual: linear,
        };
        let mut model = model_from_parts(
            Chronos2Config::from_config_json(CONFIG).unwrap(),
            embedding.clone(),
            Vec::new(),
            embedding,
            Vec::new(),
        );
        for horizon in [0, 13] {
            assert!(matches!(model.predict(&[1.0], horizon),
                Err(InferenceError::InvalidInput(message))
                if message == format!("Chronos-2: horizon {horizon} is outside 1..=12")));
        }
        model.config.max_output_patches = usize::MAX / model.config.patch_size + 1;
        assert!(
            matches!(model.predict(&[1.0], 1), Err(InferenceError::InvalidInput(message))
                if message == HORIZON_OVERFLOW_MESSAGE),
            "predict must refuse a direct overflowing config before computation"
        );
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

    #[test]
    fn request_extents_are_refused_before_allocation() {
        fn model(patch: usize, max_patches: usize) -> Chronos2 {
            let mut value: Value = serde_json::from_str(CONFIG).unwrap();
            for name in ["d_model", "d_kv", "d_ff"] {
                value[name] = 2.into();
            }
            value["num_heads"] = 1.into();
            value["num_layers"] = 0.into();
            for name in [
                "input_patch_size",
                "input_patch_stride",
                "output_patch_size",
            ] {
                value["chronos_config"][name] = patch.into();
            }
            value["chronos_config"]["max_output_patches"] = max_patches.into();
            value["chronos_config"]["quantiles"] = serde_json::json!([0.5]);
            value["chronos_config"]["use_reg_token"] = false.into();
            let config = Chronos2Config::from_config_json(&value.to_string()).unwrap();
            let linear = |input, output| Linear {
                weight: vec![0.0; input * output],
                bias: Some(vec![0.0; output]),
                in_dim: input,
                out_dim: output,
            };
            model_from_parts(
                config,
                ResidualBlock {
                    hidden: linear(3 * patch, 2),
                    output: linear(2, 2),
                    residual: linear(3 * patch, 2),
                },
                vec![1.0; 2],
                ResidualBlock {
                    hidden: linear(2, 2),
                    output: linear(2, patch),
                    residual: linear(2, patch),
                },
                vec![1.0],
            )
        }
        fn refuses<T: std::fmt::Debug>(
            call: impl FnOnce() -> Result<T, InferenceError> + std::panic::UnwindSafe,
            expected: &str,
        ) {
            let result = std::panic::catch_unwind(call)
                .expect("request extents must return InvalidInput before allocation, not panic");
            assert!(
                matches!(&result, Err(InferenceError::InvalidInput(message)) if message == expected),
                "wrong request-extent refusal: {result:?}"
            );
        }
        let ordinary = model(4, 3);
        assert_eq!(ordinary.predict(&[2.0], 3).unwrap().values, vec![2.0; 3]);
        let batch = [Chronos2Series {
            context: vec![2.0],
            ..Default::default()
        }];
        assert_eq!(ordinary.forward(&batch, 1).unwrap()[0].values, vec![2.0; 4]);
        let (forecasts, trace) = ordinary.forward_with_trace(&batch, 1).unwrap();
        assert_eq!(forecasts[0].values, vec![2.0; 4]);
        assert_eq!(trace.embeddings.len(), 4);
        assert_eq!(trace.final_state.len(), 4);
        let byte_limit = isize::MAX as usize / std::mem::size_of::<f32>();
        for (patch, patches, expected) in [
            (
                4,
                usize::MAX / 4,
                "Chronos-2: request future patch inputs exceeds usize::MAX",
            ),
            (
                4,
                byte_limit / 12 + 1,
                "Chronos-2: request future patch inputs exceeds isize::MAX bytes",
            ),
            (
                1,
                usize::MAX,
                "Chronos-2: request sequence length exceeds usize::MAX",
            ),
        ] {
            // Check the wrapped legacy element count without allocating that buffer.
            let legacy_elements = patches.wrapping_mul(3).wrapping_mul(patch);
            assert!(
                legacy_elements > byte_limit,
                "unsafe baseline witness premise"
            );
            let model = model(patch, usize::MAX / patch);
            let horizon = patches.checked_mul(patch).unwrap();
            assert!(horizon <= model.config.checked_max_horizon().unwrap());
            refuses(|| model.predict(&[2.0], horizon), expected);
            refuses(|| model.forward(&batch, patches), expected);
            refuses(|| model.forward_with_trace(&batch, patches), expected);
        }
    }

    #[test]
    fn admitted_request_extents_match_small_padded_batch() {
        let cfg = Chronos2Config::from_config_json(CONFIG).unwrap();
        let extents = checked_request_extents(&cfg, 2, 5, 2, true).unwrap();
        assert_eq!(
            (
                extents.horizon_steps,
                extents.context_patch_count,
                extents.padded_context_steps,
                extents.sequence_position_count
            ),
            (8, 2, 8, 5)
        );
        assert_eq!(
            (
                extents.context_row_count,
                extents.future_row_count,
                extents.encoder_row_count
            ),
            (4, 4, 10)
        );
        assert_eq!(
            (
                extents.context_input_elements,
                extents.future_input_elements,
                extents.encoder_hidden_elements,
                extents.future_embedding_elements,
                extents.forecast_value_elements
            ),
            (48, 48, 80, 32, 24)
        );
    }

    #[test]
    fn request_extent_checks_distinguish_elements_from_bytes() {
        let limit = isize::MAX as usize / std::mem::size_of::<f32>();
        assert_eq!(
            request_buffer_elements::<f32>("fixture", &[limit]).unwrap(),
            limit
        );
        assert!(
            matches!(request_buffer_elements::<f32>("fixture", &[limit + 1]),
            Err(InferenceError::InvalidInput(message)) if message == "Chronos-2: request fixture exceeds isize::MAX bytes")
        );
        assert!(matches!(request_product("fixture", &[usize::MAX, 2]),
            Err(InferenceError::InvalidInput(message)) if message == "Chronos-2: request fixture exceeds usize::MAX"));
    }

    #[test]
    fn request_admission_covers_transitive_buffers_without_allocating() {
        fn refuses(cfg: &Chronos2Config, patches: usize, name: &str) {
            let result = checked_request_extents(cfg, 1, 1, patches, true);
            assert!(
                matches!(result, Err(InferenceError::InvalidInput(ref message))
                if message == &format!("Chronos-2: request {name} exceeds isize::MAX bytes")),
                "{result:?}"
            );
        }
        let mut cfg = Chronos2Config::from_config_json(CONFIG).unwrap();
        cfg.patch_size = 1;
        cfg.d_model = 1;
        cfg.d_ff = 1;
        cfg.d_kv = 2;
        cfg.num_heads = 1;
        cfg.use_reg_token = false;
        cfg.quantiles = vec![0.5];
        let patches = 1usize << (usize::BITS / 2 - 1);
        cfg.max_output_patches = patches;
        refuses(&cfg, patches, "time attention scores");
        assert!(
            checked_request_extents(&cfg, 1, 1, patches, false).is_ok(),
            "a block-free model does not allocate attention scores"
        );

        let limit = isize::MAX as usize / std::mem::size_of::<f32>();
        cfg.max_output_patches = 1;
        cfg.d_ff = limit / 2 + 1;
        refuses(&cfg, 1, "feed-forward hidden");
        cfg.d_ff = 1;
        cfg.num_heads = limit / 4 + 1;
        refuses(&cfg, 1, "attention projections");
        cfg.num_heads = 1;
        cfg.d_kv = limit / 2 + 1;
        refuses(&cfg, 1, "rotary tables");

        cfg.d_kv = 2;
        cfg.quantiles = vec![0.5; 5];
        cfg.max_output_patches = limit / 4 + 1;
        let result = checked_request_extents(&cfg, 1, 1, cfg.max_output_patches, false);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(ref message))
            if message == "Chronos-2: request output patch predictions exceeds isize::MAX bytes"),
            "{result:?}"
        );
    }

    fn request_memory_fixture(block_count: usize) -> Chronos2 {
        let mut value: Value = serde_json::from_str(CONFIG).unwrap();
        for name in ["d_model", "d_kv", "d_ff"] {
            value[name] = 2.into();
        }
        value["num_heads"] = 1.into();
        value["num_layers"] = block_count.into();
        value["chronos_config"]["quantiles"] = serde_json::json!([0.5]);
        value["chronos_config"]["use_reg_token"] = false.into();
        let config = Chronos2Config::from_config_json(&value.to_string()).unwrap();
        let linear = |input, output| Linear {
            weight: vec![0.0; input * output],
            bias: Some(vec![0.0; output]),
            in_dim: input,
            out_dim: output,
        };
        let attention = || Attention {
            q: linear(2, 2),
            k: linear(2, 2),
            v: linear(2, 2),
            o: linear(2, 2),
        };
        let mut model = model_from_parts(
            config,
            ResidualBlock {
                hidden: linear(12, 2),
                output: linear(2, 2),
                residual: linear(12, 2),
            },
            vec![1.0; 2],
            ResidualBlock {
                hidden: linear(2, 2),
                output: linear(2, 4),
                residual: linear(2, 4),
            },
            vec![1.0],
        );
        model.blocks = (0..block_count)
            .map(|_| Block {
                time_norm: vec![1.0; 2],
                time_attn: attention(),
                group_norm: vec![1.0; 2],
                group_attn: attention(),
                ff_norm: vec![1.0; 2],
                wi: linear(2, 2),
                wo: linear(2, 2),
            })
            .collect();
        model
    }

    fn assert_request_memory_refusal<T: std::fmt::Debug>(
        result: Result<T, InferenceError>,
        total: usize,
        budget: usize,
    ) {
        let expected = format!(
            "Chronos-2: request memory requires {total} bytes, exceeding budget {budget} bytes"
        );
        assert!(
            matches!(&result, Err(InferenceError::InvalidInput(message)) if message == &expected),
            "expected request memory refusal {expected:?}, got {result:?}"
        );
    }

    #[test]
    fn request_memory_budget_default_and_disable_preserve_outputs() {
        let mut model = request_memory_fixture(0);
        let batch = [Chronos2Series {
            context: vec![2.0],
            ..Default::default()
        }];
        for disable in [false, true] {
            if disable {
                model.set_request_memory_budget(Some(0));
                model.set_request_memory_budget(None);
            }
            assert_eq!(model.forward(&batch, 1).unwrap()[0].values, vec![2.0; 4]);
            let (forecasts, trace) = model.forward_with_trace(&batch, 1).unwrap();
            assert_eq!(forecasts[0].values, vec![2.0; 4]);
            assert_eq!(trace.embeddings, vec![0.0; 4]);
            assert!(trace.blocks.is_empty());
            assert_eq!(trace.final_state, vec![0.0; 4]);
            let forecast = model.predict(&[2.0], 3).unwrap();
            assert_eq!(forecast.horizon, 3);
            assert_eq!(forecast.values, vec![2.0; 3]);
        }
    }

    #[test]
    fn request_memory_budget_batch_exact_boundary() {
        let mut model = request_memory_fixture(0);
        let batch = vec![
            Chronos2Series {
                context: vec![2.0],
                ..Default::default()
            };
            2
        ];
        let patch_inputs = 192; // Two context and two future rows of 12 f32 values.
        let key_mask = 16; // Four encoder positions.
        let locations = 16; // A location/scale pair for each series.
        let embeddings = 64; // Context, future and combined encoder payloads.
        let rotary_tables = 32; // Cosine and sine for two positions of width two.
        let groups = 16; // One u64 group id per series.
        let live_run = patch_inputs + key_mask + locations + embeddings + rotary_tables + groups;
        let future_encoder = 16; // One future row of width two per series.
        let projection_hidden = 16; // Two output-projection hidden rows.
        let predictions = 32; // Four f32 forecast values per series.
        let output_projection = live_run + future_encoder + projection_hidden + 2 * predictions;
        let output_construction = live_run
            + future_encoder
            + 2 * predictions
            + batch.len() * std::mem::size_of::<QuantileForecast>();
        let total = output_projection.max(output_construction);
        model.set_request_memory_budget(Some(total));
        let forecasts = model.forward(&batch, 1).unwrap();
        assert_eq!(forecasts.len(), 2);
        for forecast in forecasts {
            assert_eq!(forecast.horizon, 4);
            assert_eq!(forecast.values, vec![2.0; 4]);
        }
        model.set_request_memory_budget(Some(total - 1));
        assert_request_memory_refusal(model.forward(&batch, 1), total, total - 1);
    }

    #[test]
    fn request_memory_budget_trace_retains_actual_blocks() {
        let mut model = request_memory_fixture(2);
        let batch = [Chronos2Series {
            context: vec![2.0],
            ..Default::default()
        }];
        let patch_inputs = 96; // One context and one future row of 12 f32 values.
        let key_mask = 8; // Two encoder positions.
        let locations = 8; // One location/scale pair.
        let embeddings = 32; // Context, future and combined encoder payloads.
        let rotary_tables = 32; // Cosine and sine for two positions of width two.
        let groups = 8; // One u64 group id.
        let live_run = patch_inputs + key_mask + locations + embeddings + rotary_tables + groups;
        let attention_copies = 32; // Normalized encoder input and projected attention output.
        let attention_projections = 64; // Query, key, value and attended rows.
        let head_rows = 48; // Query, key and value rows for one attention head.
        let attention_scores = 16; // A two-by-two score matrix.
        let attention = attention_copies + attention_projections + head_rows + attention_scores;
        let untraced_total = live_run + attention;
        model.set_request_memory_budget(Some(untraced_total));
        assert_eq!(model.forward(&batch, 1).unwrap()[0].values, vec![2.0; 4]);
        model.set_request_memory_budget(Some(untraced_total - 1));
        assert_request_memory_refusal(model.forward(&batch, 1), untraced_total, untraced_total - 1);
        let descriptor = std::mem::size_of::<Vec<f32>>();
        let retained_before_last_block = 32; // Embedding and first-block snapshots.
        let final_snapshots = 64; // Embedding, both blocks and final normalized state.
        let future_encoder = 8; // One future row of width two.
        let projection_hidden = 8; // One output-projection hidden row.
        let predictions = 16; // Four f32 forecast values.
        let traced_attention = live_run + retained_before_last_block + descriptor + attention;
        let final_trace = final_snapshots + model.blocks.len() * descriptor;
        let output_projection =
            live_run + final_trace + future_encoder + projection_hidden + 2 * predictions;
        let output_construction = live_run
            + final_trace
            + future_encoder
            + 2 * predictions
            + std::mem::size_of::<QuantileForecast>();
        let total = traced_attention
            .max(output_projection)
            .max(output_construction);
        model.set_request_memory_budget(Some(total - 1));
        assert_request_memory_refusal(model.forward_with_trace(&batch, 1), total, total - 1);
        model.set_request_memory_budget(Some(total));
        let (forecasts, trace) = model.forward_with_trace(&batch, 1).unwrap();
        assert_eq!(forecasts.len(), 1);
        assert_eq!(forecasts[0].values, vec![2.0; 4]);
        assert_eq!(trace.blocks.len(), 2);
        for state in std::iter::once(&trace.embeddings)
            .chain(&trace.blocks)
            .chain(std::iter::once(&trace.final_state))
        {
            assert_eq!(state.len(), 4);
            assert!(state.iter().all(|value| value.is_finite()));
            assert_eq!(state.as_slice(), &[0.0; 4]);
        }
    }

    #[test]
    fn request_memory_budget_trace_descriptor_charge() {
        let mut config = request_memory_fixture(2).config;
        let block_count = 2;
        let descriptor = std::mem::size_of::<Vec<f32>>();
        let live_run = 184; // Patch inputs, metadata, retained embeddings and rotary tables.
        let retained_before_last_block = 32; // Embedding and first-block snapshots.
        let attention = 160; // Encoder copies, Q/K/V/attended rows, head rows and scores.
        let encoder_numeric = live_run + retained_before_last_block + attention;
        let extents = checked_request_extents(&config, 1, 1, 1, true).unwrap();
        assert_eq!(
            request_memory_peak_bytes(&config, &extents, 1, 1, block_count, true, None),
            Some(encoder_numeric + descriptor),
            "the second block retains one heap descriptor for its preceding snapshot"
        );

        config.quantiles = vec![0.1, 0.5, 0.9];
        let final_snapshots = 64; // Embedding, both blocks and final normalized state.
        let future_encoder = 8; // One future row of width two.
        let predictions_and_values = 96; // Two copies of three quantiles of four f32 values.
        let output_numeric = live_run + final_snapshots + future_encoder + predictions_and_values;
        let extents = checked_request_extents(&config, 1, 1, 1, true).unwrap();
        assert_eq!(
            request_memory_peak_bytes(&config, &extents, 1, 1, block_count, true, None),
            Some(
                output_numeric + std::mem::size_of::<QuantileForecast>() + block_count * descriptor
            ),
            "forecast construction retains a heap descriptor for every block snapshot"
        );
    }

    #[test]
    fn request_memory_budget_predict_counts_full_context_copy() {
        let mut model = request_memory_fixture(0);
        model.config.context_length = 1;
        let context = [2.0; 64];
        let batch = [Chronos2Series {
            context: context.to_vec(),
            ..Default::default()
        }];
        let patch_inputs = 96; // One context and one future row of 12 f32 values.
        let key_mask = 8; // Two encoder positions.
        let locations = 8; // One location/scale pair.
        let embeddings = 32; // Context, future and combined encoder payloads.
        let rotary_tables = 32; // Cosine and sine for two positions of width two.
        let groups = 8; // One u64 group id.
        let live_run = patch_inputs + key_mask + locations + embeddings + rotary_tables + groups;
        let future_encoder = 8; // One future row of width two.
        let projection_hidden = 8; // One output-projection hidden row.
        let predictions = 16; // Four f32 forecast values.
        let output_projection = live_run + future_encoder + projection_hidden + 2 * predictions;
        let output_construction =
            live_run + future_encoder + 2 * predictions + std::mem::size_of::<QuantileForecast>();
        let run_total = output_projection.max(output_construction);
        let copied_context = context.len() * std::mem::size_of::<f32>();
        let total = run_total + copied_context;
        let truncated_context = model.config.context_length * std::mem::size_of::<f32>();
        model.set_request_memory_budget(Some(run_total + truncated_context));
        assert_eq!(model.forward(&batch, 1).unwrap()[0].values, vec![2.0; 4]);
        assert_request_memory_refusal(
            model.predict(&context, 3),
            total,
            run_total + truncated_context,
        );
        model.set_request_memory_budget(Some(total));
        let forecast = model.predict(&context, 3).unwrap();
        assert_eq!(forecast.horizon, 3);
        assert_eq!(forecast.values, vec![2.0; 3]);
    }

    #[test]
    fn request_memory_budget_preserves_validation_and_empty_batch() {
        let mut model = request_memory_fixture(0);
        model.set_request_memory_budget(Some(0));
        assert!(model.forward(&[], 0).unwrap().is_empty());
        let (forecasts, trace) = model.forward_with_trace(&[], 0).unwrap();
        assert!(forecasts.is_empty());
        assert!(trace.embeddings.is_empty());
        assert!(trace.blocks.is_empty());
        assert!(trace.final_state.is_empty());
        assert!(
            matches!(model.predict(&[2.0], 0), Err(InferenceError::InvalidInput(message))
            if message == "Chronos-2: horizon 0 is outside 1..=12")
        );
        let contexts_error =
            "Chronos-2: contexts must be non-empty and of equal length; pad on the left with NaN";
        assert!(
            matches!(model.predict(&[], 1), Err(InferenceError::InvalidInput(message))
            if message == contexts_error)
        );
        let mut batch = vec![Chronos2Series {
            context: vec![2.0],
            ..Default::default()
        }];
        batch.push(Chronos2Series {
            context: vec![2.0; 2],
            ..Default::default()
        });
        assert!(
            matches!(model.forward(&batch, 1), Err(InferenceError::InvalidInput(message))
            if message == contexts_error)
        );
        batch.pop();
        batch[0].future_covariates = Some(vec![2.0; 5]);
        assert!(
            matches!(model.forward(&batch, 1), Err(InferenceError::InvalidInput(message))
            if message == "Chronos-2: 5 future covariate values exceed the 4-step horizon")
        );
        batch[0].future_covariates = None;
        model.config.max_output_patches = usize::MAX / 4;
        assert!(
            matches!(model.forward(&batch, usize::MAX / 4), Err(InferenceError::InvalidInput(message))
            if message == "Chronos-2: request future patch inputs exceeds usize::MAX")
        );
    }

    #[test]
    fn request_memory_budget_accounting_overflow_is_optional() {
        let mut model = request_memory_fixture(0);
        let limit = isize::MAX as usize / std::mem::size_of::<f32>();
        model.config.d_model = limit / 2;
        let extents = checked_request_extents(&model.config, 1, 1, 1, false)
            .expect("each huge-D buffer remains representable before aggregate accounting");
        assert!(
            model
                .check_request_memory(&extents, 1, 1, false, None)
                .is_ok()
        );
        model.set_request_memory_budget(Some(usize::MAX));
        assert!(
            matches!(model.check_request_memory(&extents, 1, 1, false, None),
            Err(InferenceError::InvalidInput(message))
            if message == "Chronos-2: request memory accounting exceeds usize::MAX")
        );
    }
}
