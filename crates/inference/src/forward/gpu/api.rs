use std::fmt;

/// **Unstable**: GPU forward result type; error type may gain new variants.
pub type Result<T> = std::result::Result<T, GpuForwardError>;

/// **Unstable**: GPU forward pass error; variants may expand with new GPU backends.
#[derive(Debug)]
pub enum GpuForwardError {
    NoAdapter,
    RequestDevice(wgpu::RequestDeviceError),
    InvalidInput(String),
    InvalidWeights(String),
    Limit(String),
    BufferMap(wgpu::BufferAsyncError),
    ChannelClosed,
    ExecutionLockPoisoned,
}

impl fmt::Display for GpuForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no suitable GPU adapter found"),
            Self::RequestDevice(e) => write!(f, "failed to request GPU device: {e}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::InvalidWeights(msg) => write!(f, "invalid weights: {msg}"),
            Self::Limit(msg) => write!(f, "GPU limit exceeded: {msg}"),
            Self::BufferMap(e) => write!(f, "failed to map readback buffer: {e}"),
            Self::ChannelClosed => {
                write!(f, "map_async callback channel closed unexpectedly")
            }
            Self::ExecutionLockPoisoned => write!(f, "execution lock was poisoned"),
        }
    }
}

impl std::error::Error for GpuForwardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RequestDevice(e) => Some(e),
            Self::BufferMap(e) => Some(e),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for GpuForwardError {
    fn from(value: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(value)
    }
}

impl From<wgpu::BufferAsyncError> for GpuForwardError {
    fn from(value: wgpu::BufferAsyncError) -> Self {
        Self::BufferMap(value)
    }
}

/// **Unstable**: Qwen3 GPU model configuration; fields mirror CPU config but may diverge.
#[derive(Clone, Debug)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

impl Qwen3Config {
    /// **Unstable**: Q projection dimension helper.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// **Unstable**: KV projection dimension helper.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// **Unstable**: GQA group count helper.
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// **Unstable**: GPU runtime configuration; max_seq_len and embedding upload flag may change.
#[derive(Clone, Debug)]
pub struct GpuRuntimeConfig {
    /// Practical inference-time ceiling for buffer allocation.
    pub max_seq_len: usize,
    /// If true, upload the embedding table to GPU when limits allow it.
    pub upload_embeddings_to_gpu: bool,
}

impl Default for GpuRuntimeConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            upload_embeddings_to_gpu: false,
        }
    }
}

/// **Unstable**: GPU per-layer weights; field set mirrors transformer layer and may change.
#[derive(Clone, Debug)]
pub struct Qwen3LayerWeights {
    pub q_proj_weight: Vec<f32>,
    pub k_proj_weight: Vec<f32>,
    pub v_proj_weight: Vec<f32>,
    pub o_proj_weight: Vec<f32>,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub input_layernorm_weight: Vec<f32>,
    pub gate_proj_weight: Vec<f32>,
    pub up_proj_weight: Vec<f32>,
    pub down_proj_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
}

/// **Unstable**: GPU full model weights; structure will change when embedding GPU upload is wired.
#[derive(Clone, Debug)]
pub struct Qwen3Weights {
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<Qwen3LayerWeights>,
    pub norm_weight: Vec<f32>,
}

impl Qwen3Weights {
    /// **Unstable**: validate weight tensor dimensions against config.
    pub fn validate(&self, config: &Qwen3Config) -> Result<()> {
        if self.layers.len() != config.num_hidden_layers {
            return Err(GpuForwardError::InvalidWeights(format!(
                "expected {} layers, got {}",
                config.num_hidden_layers,
                self.layers.len()
            )));
        }

        let hidden = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let head_dim = config.head_dim;
        let intermediate = config.intermediate_size;
        let embed_expected = checked_mul(config.vocab_size, hidden, "embed_tokens")?;

        if self.embed_tokens.len() != embed_expected {
            return Err(GpuForwardError::InvalidWeights(format!(
                "embed_tokens length mismatch: expected {embed_expected}, got {}",
                self.embed_tokens.len()
            )));
        }
        if self.norm_weight.len() != hidden {
            return Err(GpuForwardError::InvalidWeights(format!(
                "norm_weight length mismatch: expected {hidden}, got {}",
                self.norm_weight.len()
            )));
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            check_len(
                &layer.q_proj_weight,
                checked_mul(q_dim, hidden, "q_proj")?,
                idx,
                "q_proj_weight",
            )?;
            check_len(
                &layer.k_proj_weight,
                checked_mul(kv_dim, hidden, "k_proj")?,
                idx,
                "k_proj_weight",
            )?;
            check_len(
                &layer.v_proj_weight,
                checked_mul(kv_dim, hidden, "v_proj")?,
                idx,
                "v_proj_weight",
            )?;
            check_len(
                &layer.o_proj_weight,
                checked_mul(hidden, q_dim, "o_proj")?,
                idx,
                "o_proj_weight",
            )?;
            check_len(&layer.q_norm_weight, head_dim, idx, "q_norm_weight")?;
            check_len(&layer.k_norm_weight, head_dim, idx, "k_norm_weight")?;
            check_len(
                &layer.input_layernorm_weight,
                hidden,
                idx,
                "input_layernorm_weight",
            )?;
            check_len(
                &layer.gate_proj_weight,
                checked_mul(intermediate, hidden, "gate_proj")?,
                idx,
                "gate_proj_weight",
            )?;
            check_len(
                &layer.up_proj_weight,
                checked_mul(intermediate, hidden, "up_proj")?,
                idx,
                "up_proj_weight",
            )?;
            check_len(
                &layer.down_proj_weight,
                checked_mul(hidden, intermediate, "down_proj")?,
                idx,
                "down_proj_weight",
            )?;
            check_len(
                &layer.post_attention_layernorm_weight,
                hidden,
                idx,
                "post_attention_layernorm_weight",
            )?;
        }

        Ok(())
    }
}

pub(super) fn check_len(slice: &[f32], expected: usize, layer: usize, name: &str) -> Result<()> {
    if slice.len() != expected {
        return Err(GpuForwardError::InvalidWeights(format!(
            "layer {layer} {name} length mismatch: expected {expected}, got {}",
            slice.len()
        )));
    }
    Ok(())
}

pub(super) fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| GpuForwardError::Limit(format!("overflow while computing {what}")))
}
