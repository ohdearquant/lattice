//! Qwen3.5 shape-checked owned tensor loading, required-tensor validation, layer-specific weight loaders, and `load_weights`.
use super::weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights, ModelWeights, MoeLayerWeights, MoeRouter, RoutedExperts,
    SharedExpert,
};
use crate::attention::gdn::GatedDeltaNetWeights;
use lattice_arch::qwen35_config::{Qwen35Config, ValidatedQwen35Config};
use lattice_arch::qwen35_tensors::{
    FLAT_MODEL_EMBED, FLAT_MODEL_PREFIX, LANGUAGE_MODEL_EMBED, LANGUAGE_MODEL_PREFIX,
    qwen_layer_tensor_prefix_with_prefix, qwen_required_tensor_names_with_prefix,
    resolve_tensor_prefix,
};
use lattice_compute::forward::cpu::weight_matrix::{CpuWeightStore, WeightMatrix};
use lattice_inference_core::InferenceError;
use lattice_weights::{Bf16Bits, TensorSource};
use std::collections::BTreeMap;

fn source_tensor_prefix<T: TensorSource + ?Sized>(
    source: &mut T,
) -> Result<&'static str, InferenceError> {
    let has_language_model = source.has_tensor(LANGUAGE_MODEL_EMBED)?;
    let has_flat_model = source.has_tensor(FLAT_MODEL_EMBED)?;
    resolve_tensor_prefix(has_language_model, has_flat_model)
}

/// CLASS C ingress: takes `&ValidatedQwen35Config` rather than a raw `&Qwen35Config`, so an
/// unvalidated config cannot reach the required-tensor-name derivation for a checkpoint on
/// this (non-Metal) load path. See the CLASS C ingress table in the PR body.
pub(super) fn validate_required_tensor_names<T: TensorSource + ?Sized>(
    source: &mut T,
    validated_cfg: &ValidatedQwen35Config,
) -> Result<(), InferenceError> {
    let cfg: &Qwen35Config = validated_cfg;
    let prefix = source_tensor_prefix(source)?;
    for name in qwen_required_tensor_names_with_prefix(cfg, prefix) {
        if !source.has_tensor(&name)? {
            return Err(InferenceError::MissingTensor(name));
        }
    }
    Ok(())
}

/// Loads and shape-checks a tensor, validating the declared shape against
/// `expected` *before* materializing the owned buffer. Rejecting a mismatch
/// at the metadata stage (via [`TensorSource::tensor_shape`]) avoids decoding
/// and copying a tensor the caller is about to discard, which matters for a
/// checkpoint carrying an oversized mismatched tensor. A tensor absent from
/// the source (`tensor_shape` returns `Ok(None)`) falls through to
/// `get_f32_tensor_owned`, which reports the missing-tensor error.
fn load_owned_tensor_checked<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    check_declared_shape(source, name, expected)?;
    let (data, shape) = source.get_f32_tensor_owned(name)?;
    if shape != expected {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: shape,
        });
    }
    Ok(data)
}

/// Rejects a header-declared shape that differs from `expected` before any tensor data is
/// materialized. A source that cannot report a shape up front returns `None` and passes here;
/// the caller checks the loaded shape afterwards, but any source that can answer rejects a
/// mismatch before the (potentially huge) allocation and copy happen.
fn check_declared_shape<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<(), InferenceError> {
    if let Some(declared) = source.tensor_shape(name)?
        && declared != expected
    {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: declared,
        });
    }
    Ok(())
}

/// Loads and shape-checks a weight matrix in the given store. Under
/// [`CpuWeightStore::Bf16`] a tensor the checkpoint stores as BF16 is held as its bits;
/// any other dtype, and every tensor under [`CpuWeightStore::F32`], is loaded as f32 by
/// [`load_owned_tensor_checked`].
fn load_matrix_checked<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
    store: CpuWeightStore,
) -> Result<WeightMatrix, InferenceError> {
    if store == CpuWeightStore::Bf16 {
        check_declared_shape(source, name, expected)?;
        if let Some((bits, shape)) = source.get_bf16_bits_owned(name)? {
            if shape != expected {
                return Err(InferenceError::ShapeMismatch {
                    name: name.to_string(),
                    expected: expected.to_vec(),
                    actual: shape,
                });
            }
            return Ok(WeightMatrix::Bf16(bits));
        }
    }
    Ok(load_owned_tensor_checked(source, name, expected)?.into())
}

/// The tensors below each layer prefix that a dense load reads through [`load_matrix_checked`].
const BF16_STORE_LAYER_MATRICES: [&str; 10] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// **Unstable**: whether a dense checkpoint loaded in [`CpuWeightStore::Bf16`] holds the
/// tensor `name` as bf16 bits when the checkpoint stores it as BF16: the token embedding, the
/// output head when the config unties it, and each layer's attention, GatedDeltaNet and
/// feed-forward projection matrices. Every other tensor is held as f32 in either store, so a
/// memory estimate for the bf16 store counts only these at their on-disk size.
pub fn bf16_store_holds_as_bits(name: &str) -> bool {
    if name == LANGUAGE_MODEL_EMBED || name == FLAT_MODEL_EMBED || name == "lm_head.weight" {
        return true;
    }
    [LANGUAGE_MODEL_PREFIX, FLAT_MODEL_PREFIX]
        .iter()
        .any(|prefix| {
            name.strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix("layers."))
                .and_then(|rest| rest.split_once('.'))
                .is_some_and(|(index, tensor)| {
                    !index.is_empty()
                        && index.bytes().all(|byte| byte.is_ascii_digit())
                        && BF16_STORE_LAYER_MATRICES.contains(&tensor)
                })
        })
}

/// **Unstable**: how a CPU load held the tensors it read, counted by each tensor's
/// header-declared dtype.
///
/// [`Display`](std::fmt::Display) renders the one line a server prints at startup, so a
/// checkpoint that stores nothing as BF16 says so when the bf16 store was asked for instead
/// of silently staying f32.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightLoadReport {
    store: CpuWeightStore,
    counts: BTreeMap<(String, CpuWeightStore), usize>,
}

impl WeightLoadReport {
    fn new(store: CpuWeightStore) -> Self {
        Self {
            store,
            counts: BTreeMap::new(),
        }
    }

    fn record(&mut self, dtype: Option<String>, held: CpuWeightStore) {
        let dtype = dtype.unwrap_or_else(|| "unknown".to_string());
        *self.counts.entry((dtype, held)).or_insert(0) += 1;
    }

    /// **Unstable**: the store the load was asked for.
    pub fn store(&self) -> CpuWeightStore {
        self.store
    }

    /// **Unstable**: tensors declared as `dtype` that are held in `held`. A tensor whose
    /// source could not report a dtype is counted under `unknown`.
    pub fn count(&self, dtype: &str, held: CpuWeightStore) -> usize {
        self.counts
            .get(&(dtype.to_string(), held))
            .copied()
            .unwrap_or(0)
    }

    /// **Unstable**: tensors held in `held`, whatever their declared dtype.
    pub fn held_as(&self, held: CpuWeightStore) -> usize {
        self.counts
            .iter()
            .filter(|((_, store), _)| *store == held)
            .map(|(_, count)| count)
            .sum()
    }
}

impl std::fmt::Display for WeightLoadReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cpu weight store {}:", self.store)?;
        if self.counts.is_empty() {
            f.write_str(" no tensors read")?;
        }
        for (index, ((dtype, held), count)) in self.counts.iter().enumerate() {
            let separator = if index == 0 { " " } else { ", " };
            write!(f, "{separator}{count} {dtype} held as {held}")?;
        }
        if self.store == CpuWeightStore::Bf16 && self.held_as(CpuWeightStore::Bf16) == 0 {
            f.write_str("; the checkpoint stores no matrix as BF16, so every matrix stays f32")?;
        }
        Ok(())
    }
}

/// A [`TensorSource`] that counts each tensor read through it into a [`WeightLoadReport`].
struct RecordingSource<'a, T: ?Sized> {
    inner: &'a mut T,
    report: WeightLoadReport,
}

impl<T: TensorSource + ?Sized> TensorSource for RecordingSource<'_, T> {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        self.inner.has_tensor(name)
    }

    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        self.inner.tensor_shape(name)
    }

    fn tensor_dtype(&mut self, name: &str) -> Result<Option<String>, InferenceError> {
        self.inner.tensor_dtype(name)
    }

    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        let tensor = self.inner.get_f32_tensor_owned(name)?;
        let dtype = self.inner.tensor_dtype(name)?;
        self.report.record(dtype, CpuWeightStore::F32);
        Ok(tensor)
    }

    fn get_bf16_bits_owned(&mut self, name: &str) -> Result<Option<Bf16Bits>, InferenceError> {
        let bits = self.inner.get_bf16_bits_owned(name)?;
        if bits.is_some() {
            let dtype = self.inner.tensor_dtype(name)?;
            self.report.record(dtype, CpuWeightStore::Bf16);
        }
        Ok(bits)
    }
}

/// Doubles `value`, returning a typed error instead of panicking (debug) or wrapping
/// (release) when the config-derived dimension is too large for `usize`.
fn checked_double(value: usize, what: &str) -> Result<usize, InferenceError> {
    value
        .checked_mul(2)
        .ok_or_else(|| InferenceError::InvalidInput(format!("{what} overflows usize: 2 * {value}")))
}

/// Full-attention Q projection row count (`num_attention_heads * head_dim`), computed
/// with checked arithmetic so an attacker-controlled config.json cannot overflow it.
fn checked_full_q_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    cfg.num_attention_heads
        .checked_mul(cfg.head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "full attention q_dim overflows usize: num_attention_heads({}) * head_dim({})",
                cfg.num_attention_heads, cfg.head_dim
            ))
        })
}

/// Full-attention KV projection row count (`num_key_value_heads * head_dim`), checked.
fn checked_full_kv_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    cfg.num_key_value_heads
        .checked_mul(cfg.head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "full attention kv_dim overflows usize: num_key_value_heads({}) * head_dim({})",
                cfg.num_key_value_heads, cfg.head_dim
            ))
        })
}

/// GatedDeltaNet combined QKV projection row count (`Q + K + V`), checked. Mirrors
/// `Qwen35Config::linear_qkv_dim`'s formula but rejects overflow with a typed error
/// instead of panicking (debug) or wrapping (release).
fn checked_linear_qkv_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    let k = cfg
        .linear_num_key_heads
        .checked_mul(cfg.linear_key_head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "linear attention k dim overflows usize: linear_num_key_heads({}) * linear_key_head_dim({})",
                cfg.linear_num_key_heads, cfg.linear_key_head_dim
            ))
        })?;
    let value_heads = cfg.linear_num_value_heads();
    let v = value_heads.checked_mul(cfg.linear_value_head_dim).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "linear attention v dim overflows usize: linear_num_value_heads({}) * linear_value_head_dim({})",
            value_heads, cfg.linear_value_head_dim
        ))
    })?;
    k.checked_add(k)
        .and_then(|qk| qk.checked_add(v))
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "linear attention qkv_dim overflows usize: q({k}) + k({k}) + v({v})"
            ))
        })
}

/// GatedDeltaNet output projection row count (`linear_num_value_heads * linear_value_head_dim`), checked.
fn checked_linear_output_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    let value_heads = cfg.linear_num_value_heads();
    value_heads.checked_mul(cfg.linear_value_head_dim).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "linear attention output_dim overflows usize: linear_num_value_heads({}) * linear_value_head_dim({})",
            value_heads, cfg.linear_value_head_dim
        ))
    })
}

fn load_moe_ffn_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    prefix: &str,
    hidden: usize,
) -> Result<FeedForwardWeights, InferenceError> {
    let num_experts = cfg
        .num_experts
        .ok_or_else(|| InferenceError::InvalidInput("MoE config missing num_experts".into()))?;
    let top_k = cfg.num_experts_per_tok.ok_or_else(|| {
        InferenceError::InvalidInput("MoE config missing num_experts_per_tok".into())
    })?;
    let moe_inter = cfg.moe_intermediate_size();
    let shared_inter = cfg.shared_expert_intermediate_size();
    let doubled_moe_inter = checked_double(
        moe_inter,
        "MoE gate_up_proj row count (2 * moe_intermediate_size)",
    )?;

    let router = MoeRouter::new(
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.gate.weight"),
            &[num_experts, hidden],
        )?,
        num_experts,
        top_k,
        hidden,
    )?;

    let experts = RoutedExperts::new(
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.experts.gate_up_proj"),
            &[num_experts, doubled_moe_inter, hidden],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.experts.down_proj"),
            &[num_experts, hidden, moe_inter],
        )?,
        num_experts,
        hidden,
        moe_inter,
    )?;

    let shared_expert = SharedExpert::new(
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            &[shared_inter, hidden],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            &[shared_inter, hidden],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            &[hidden, shared_inter],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert_gate.weight"),
            &[1, hidden],
        )?,
        hidden,
        shared_inter,
    )?;

    Ok(FeedForwardWeights::Moe(MoeLayerWeights {
        router,
        experts,
        shared_expert,
    }))
}

fn load_dense_ffn_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    prefix: &str,
    store: CpuWeightStore,
) -> Result<FeedForwardWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    // Checked against the declared safetensors axis shape (not just the flattened
    // element count) so a transposed tensor with the same element count — which would
    // otherwise be silently reinterpreted as row-major — is rejected here.
    let gp = load_matrix_checked(
        source,
        &format!("{prefix}.mlp.gate_proj.weight"),
        &[inter, hidden],
        store,
    )?;
    let up = load_matrix_checked(
        source,
        &format!("{prefix}.mlp.up_proj.weight"),
        &[inter, hidden],
        store,
    )?;
    let dp = load_matrix_checked(
        source,
        &format!("{prefix}.mlp.down_proj.weight"),
        &[hidden, inter],
        store,
    )?;
    Ok(FeedForwardWeights::Dense(DenseFfnWeights {
        gate_proj: gp,
        up_proj: up,
        down_proj: dp,
    }))
}

fn load_full_attention_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    prefix: &str,
    store: CpuWeightStore,
) -> Result<AttentionWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let q_dim = checked_full_q_dim(cfg)?;
    let kv_dim = checked_full_kv_dim(cfg)?;
    let doubled_q_dim = checked_double(
        q_dim,
        "full attention q_proj row count (2 * q_dim, fused sigmoid gate)",
    )?;
    let qw = load_matrix_checked(
        source,
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[doubled_q_dim, hidden],
        store,
    )?;
    let kw = load_matrix_checked(
        source,
        &format!("{prefix}.self_attn.k_proj.weight"),
        &[kv_dim, hidden],
        store,
    )?;
    let vw = load_matrix_checked(
        source,
        &format!("{prefix}.self_attn.v_proj.weight"),
        &[kv_dim, hidden],
        store,
    )?;
    let ow = load_matrix_checked(
        source,
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[hidden, q_dim],
        store,
    )?;
    let qn = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.q_norm.weight"),
        &[head_dim],
    )?;
    let kn = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.k_norm.weight"),
        &[head_dim],
    )?;
    Ok(AttentionWeights::Full(FullAttentionLayerWeights {
        q_proj: qw,
        k_proj: kw,
        v_proj: vw,
        o_proj: ow,
        q_norm: qn,
        k_norm: kn,
    }))
}

fn load_linear_attention_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    prefix: &str,
    store: CpuWeightStore,
) -> Result<AttentionWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let qkv_dim = checked_linear_qkv_dim(cfg)?;
    let output_dim = checked_linear_output_dim(cfg)?;
    let kernel_size = cfg.linear_conv_kernel_dim;
    let value_heads = cfg.linear_num_value_heads();
    let value_dim = cfg.linear_value_head_dim;
    // Checked against the declared safetensors axis shape (not just the flattened
    // element count) so a transposed tensor with the same element count — which would
    // otherwise be silently reinterpreted as row-major — is rejected here.
    let qkv = load_matrix_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
        &[qkv_dim, hidden],
        store,
    )?;
    let z = load_matrix_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_z.weight"),
        &[output_dim, hidden],
        store,
    )?;
    let b = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_b.weight"),
        &[value_heads, hidden],
    )?;
    let a = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_a.weight"),
        &[value_heads, hidden],
    )?;
    let alog = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.A_log"),
        &[value_heads],
    )?;
    let dtb = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.dt_bias"),
        &[value_heads],
    )?;
    // conv1d.weight's declared safetensors shape is [conv_dim, 1, kernel_size]; reshaped
    // to [conv_dim, kernel_size] below (conv_dim == qkv_dim for this architecture).
    let cw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.conv1d.weight"),
        &[qkv_dim, 1, kernel_size],
    )?;
    let nw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.norm.weight"),
        &[value_dim],
    )?;
    let op = load_matrix_checked(
        source,
        &format!("{prefix}.linear_attn.out_proj.weight"),
        &[hidden, output_dim],
        store,
    )?;

    // conv1d.weight is [conv_dim, 1, kernel_size] — reshape to [conv_dim, kernel_size]
    Ok(AttentionWeights::Linear(GatedDeltaNetWeights {
        in_proj_qkv: qkv,
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: z,
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: b,
        in_proj_b_rows: value_heads,
        in_proj_b_cols: hidden,
        in_proj_a: a,
        in_proj_a_rows: value_heads,
        in_proj_a_cols: hidden,
        a_log: alog,
        dt_bias: dtb,
        conv1d_weight: cw,
        conv_dim: qkv_dim,
        kernel_size,
        norm_weight: nw,
        out_proj: op,
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    }))
}

/// Load all per-model and per-layer tensors, shape-validating each one against a
/// config-derived expected shape via [`load_owned_tensor_checked`]. A checkpoint
/// tensor whose shape does not match returns `Err(InferenceError::ShapeMismatch)`
/// rather than being accepted and later panicking downstream (e.g. in a GEMM
/// bounds assertion during the forward pass).
/// CLASS C ingress: takes `&ValidatedQwen35Config` rather than a raw `&Qwen35Config`, so an
/// unvalidated config cannot reach tensor materialization on this (non-Metal) load path.
/// See the CLASS C ingress table in the PR body.
///
/// `store` decides how the projection, feed-forward, embedding and head matrices are held
/// ([`load_matrix_checked`]); the returned report counts every tensor read by its declared
/// dtype and its store. A MoE checkpoint refuses [`CpuWeightStore::Bf16`] before any tensor
/// is read: that store has not been measured on the MoE path.
pub(super) fn load_weights_with_store<T: TensorSource + ?Sized>(
    source: &mut T,
    validated_cfg: &ValidatedQwen35Config,
    store: CpuWeightStore,
) -> Result<(ModelWeights, WeightLoadReport), InferenceError> {
    let cfg: &Qwen35Config = validated_cfg;
    if store == CpuWeightStore::Bf16 && cfg.is_moe() {
        return Err(InferenceError::InvalidInput(
            "the bf16 CPU weight store is not measured for mixture-of-experts checkpoints; \
             load this checkpoint with the f32 store"
                .to_string(),
        ));
    }
    let mut recording = RecordingSource {
        inner: source,
        report: WeightLoadReport::new(store),
    };
    let weights = load_weights_in_store(&mut recording, cfg, store)?;
    Ok((weights, recording.report))
}

/// [`load_weights_with_store`] in the f32 store, without the report.
#[cfg(test)]
pub(super) fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    validated_cfg: &ValidatedQwen35Config,
) -> Result<ModelWeights, InferenceError> {
    load_weights_with_store(source, validated_cfg, CpuWeightStore::F32).map(|(weights, _)| weights)
}

fn load_weights_in_store<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    store: CpuWeightStore,
) -> Result<ModelWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let tensor_prefix = source_tensor_prefix(source)?;

    let embed_tokens = load_matrix_checked(
        source,
        &format!("{tensor_prefix}embed_tokens.weight"),
        &[cfg.vocab_size, hidden],
        store,
    )?;

    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load_matrix_checked(
            source,
            "lm_head.weight",
            &[cfg.vocab_size, hidden],
            store,
        )?)
    };

    let final_norm =
        load_owned_tensor_checked(source, &format!("{tensor_prefix}norm.weight"), &[hidden])?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let prefix = qwen_layer_tensor_prefix_with_prefix(tensor_prefix, i);

        let iln = load_owned_tensor_checked(
            source,
            &format!("{prefix}.input_layernorm.weight"),
            &[hidden],
        )?;
        let paln = load_owned_tensor_checked(
            source,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[hidden],
        )?;

        let ffn = if cfg.is_moe() {
            load_moe_ffn_weights(source, cfg, &prefix, hidden)?
        } else {
            load_dense_ffn_weights(source, cfg, &prefix, store)?
        };

        let common = CommonLayerWeights {
            input_layernorm: iln,
            post_attention_layernorm: paln,
            ffn,
        };

        let attn = if cfg.is_full_attention(i) {
            load_full_attention_weights(source, cfg, &prefix, store)?
        } else {
            load_linear_attention_weights(source, cfg, &prefix, store)?
        };

        layers.push((attn, common));
    }

    Ok(ModelWeights {
        embed_tokens,
        lm_head,
        final_norm,
        layers,
    })
}

#[cfg(test)]
mod tests;
