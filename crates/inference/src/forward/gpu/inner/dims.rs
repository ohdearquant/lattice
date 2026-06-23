use super::api::checked_mul;
use super::api::{GpuForwardError, GpuRuntimeConfig, Qwen3Config, Result};
use super::params::ELEM_WG;
use super::util::{bytes_f32, ceil_div_u32, to_u32};

#[derive(Clone, Copy)]
pub(super) struct ForwardDims {
    pub(super) seq: u32,
    pub(super) hidden: u32,
    pub(super) q_dim: u32,
    pub(super) kv_dim: u32,
    pub(super) intermediate: u32,
    pub(super) head_dim: u32,
    pub(super) num_heads: u32,
    pub(super) num_kv_heads: u32,
    pub(super) groups: u32,
    pub(super) hidden_elems: u32,
    pub(super) intermediate_elems: u32,
    pub(super) seq_heads: u32,
    pub(super) seq_kv_heads: u32,
}

impl ForwardDims {
    pub(super) fn from_config(config: &Qwen3Config, seq_len: usize) -> Result<Self> {
        let seq = to_u32(seq_len, "seq_len")?;
        let hidden = to_u32(config.hidden_size, "hidden_size")?;
        let q_dim = to_u32(config.q_dim(), "q_dim")?;
        let kv_dim = to_u32(config.kv_dim(), "kv_dim")?;
        let intermediate = to_u32(config.intermediate_size, "intermediate_size")?;
        let head_dim = to_u32(config.head_dim, "head_dim")?;
        let num_heads = to_u32(config.num_attention_heads, "num_attention_heads")?;
        let num_kv_heads = to_u32(config.num_key_value_heads, "num_key_value_heads")?;
        let groups = to_u32(config.num_groups(), "num_groups")?;
        let hidden_elems = to_u32(
            checked_mul(seq_len, config.hidden_size, "seq_len * hidden_size")?,
            "hidden_elems",
        )?;
        let intermediate_elems = to_u32(
            checked_mul(
                seq_len,
                config.intermediate_size,
                "seq_len * intermediate_size",
            )?,
            "intermediate_elems",
        )?;
        let seq_heads = to_u32(
            checked_mul(
                seq_len,
                config.num_attention_heads,
                "seq_len * num_attention_heads",
            )?,
            "seq_heads",
        )?;
        let seq_kv_heads = to_u32(
            checked_mul(
                seq_len,
                config.num_key_value_heads,
                "seq_len * num_key_value_heads",
            )?,
            "seq_kv_heads",
        )?;
        Ok(Self {
            seq,
            hidden,
            q_dim,
            kv_dim,
            intermediate,
            head_dim,
            num_heads,
            num_kv_heads,
            groups,
            hidden_elems,
            intermediate_elems,
            seq_heads,
            seq_kv_heads,
        })
    }
}

pub(super) fn validate_model_config(config: &Qwen3Config) -> Result<()> {
    if config.num_attention_heads == 0
        || config.num_key_value_heads == 0
        || config.hidden_size == 0
        || config.intermediate_size == 0
        || config.head_dim == 0
    {
        return Err(GpuForwardError::InvalidInput(
            "model config dimensions must be non-zero".to_string(),
        ));
    }
    if config.head_dim % 2 != 0 {
        return Err(GpuForwardError::InvalidInput(
            "head_dim must be even for RoPE".to_string(),
        ));
    }
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        return Err(GpuForwardError::InvalidInput(
            "num_attention_heads must be divisible by num_key_value_heads".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn validate_runtime_feasibility(
    limits: &wgpu::Limits,
    config: &Qwen3Config,
    runtime: &GpuRuntimeConfig,
) -> Result<()> {
    let scores_elems = checked_mul(
        checked_mul(runtime.max_seq_len, runtime.max_seq_len, "max_seq_len^2")?,
        config.num_attention_heads,
        "scores elements",
    )?;
    let scores_bytes = bytes_f32(scores_elems)?;
    let hidden_bytes = bytes_f32(checked_mul(
        runtime.max_seq_len,
        config.hidden_size,
        "hidden bytes",
    )?)?;
    let q_bytes = bytes_f32(checked_mul(runtime.max_seq_len, config.q_dim(), "q bytes")?)?;
    let kv_bytes = bytes_f32(checked_mul(
        runtime.max_seq_len,
        config.kv_dim(),
        "kv bytes",
    )?)?;
    let intermediate_bytes = bytes_f32(checked_mul(
        runtime.max_seq_len,
        config.intermediate_size,
        "intermediate bytes",
    )?)?;

    let max_storage = limits.max_storage_buffer_binding_size;
    let max_buffer = limits.max_buffer_size;
    for (name, size) in [
        ("hidden", hidden_bytes),
        ("q", q_bytes),
        ("k", kv_bytes),
        ("v", kv_bytes),
        ("attn_out", q_bytes),
        ("gate", intermediate_bytes),
        ("up", intermediate_bytes),
        ("scores", scores_bytes),
    ] {
        if size > max_buffer {
            return Err(GpuForwardError::Limit(format!(
                "{name} buffer requires {size} bytes but max_buffer_size is {max_buffer}"
            )));
        }
        if size > max_storage as u64 {
            return Err(GpuForwardError::Limit(format!(
                "{name} binding requires {size} bytes but max_storage_buffer_binding_size is {max_storage}"
            )));
        }
    }

    let max_workgroups = limits.max_compute_workgroups_per_dimension as usize;
    let q_head_rows = checked_mul(
        runtime.max_seq_len,
        config.num_attention_heads,
        "q head rows",
    )?;
    let k_head_rows = checked_mul(
        runtime.max_seq_len,
        config.num_key_value_heads,
        "k head rows",
    )?;
    if q_head_rows > max_workgroups {
        return Err(GpuForwardError::Limit(format!(
            "q-head RMSNorm requires {q_head_rows} workgroups, limit is {max_workgroups}"
        )));
    }
    if k_head_rows > max_workgroups {
        return Err(GpuForwardError::Limit(format!(
            "k-head RMSNorm requires {k_head_rows} workgroups, limit is {max_workgroups}"
        )));
    }

    let elem_groups = ceil_div_u32(
        to_u32(
            checked_mul(runtime.max_seq_len, config.intermediate_size, "elem groups")?,
            "elem groups",
        )?,
        ELEM_WG,
    ) as usize;
    if elem_groups > max_workgroups {
        return Err(GpuForwardError::Limit(format!(
            "elementwise FFN kernels require {elem_groups} workgroups, limit is {max_workgroups}"
        )));
    }

    Ok(())
}
