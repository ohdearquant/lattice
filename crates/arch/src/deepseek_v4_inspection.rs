//! Experimental root-config metadata inspection; this does not enable inference.

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

use lattice_inference_core::InferenceError;
use lattice_weights::bounded_config::{MAX_CONFIG_JSON_BYTES, read_config_json_bounded};
use serde::de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

const MAX_ITEMS: usize = 4096;
const MAX_GEOMETRY: usize = 16_777_216;

/// Parsed metadata only. It is not an executable model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Inspection {
    /// Number of base layers in the root config.
    pub num_hidden_layers: usize,
    /// The root config's `num_nextn_predict_layers`, not a DSpark stage count.
    pub num_nextn_predict_layers: usize,
    /// Selected validated metadata geometry, not an execution plan.
    pub geometry: Geometry,
    /// DSpark block size declared by the root config.
    pub dspark_block_size: usize,
    /// DSpark noise token ID declared by the root config.
    pub dspark_noise_token_id: usize,
    /// DSpark Markov rank declared by the root config.
    pub dspark_markov_rank: usize,
    /// The root compression entries, with no inferred trunk/draft split.
    pub compress_ratios: Vec<usize>,
    /// DSpark target layer IDs in their original order.
    pub dspark_target_layer_ids: Vec<usize>,
    /// Names of fields whose semantics this inspector does not validate.
    pub unsupported_fields: Vec<String>,
}

/// Selected root dimensions. These values are not evidence of model support.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    /// Hidden width.
    pub hidden_size: usize,
    /// Attention query heads.
    pub num_attention_heads: usize,
    /// Attention head width.
    pub head_dim: usize,
    /// Rotary head subdimension.
    pub qk_rope_head_dim: usize,
    /// Key/value heads.
    pub num_key_value_heads: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Declared checkpoint context maximum, not an allocated runtime capacity.
    pub max_position_embeddings: usize,
    /// Routed expert count.
    pub n_routed_experts: usize,
    /// Shared expert count.
    pub n_shared_experts: usize,
    /// Routed experts selected per token.
    pub num_experts_per_tok: usize,
    /// Expert intermediate width.
    pub moe_intermediate_size: usize,
    /// Index head width.
    pub index_head_dim: usize,
    /// Output projection groups.
    pub o_groups: usize,
}

impl Inspection {
    /// Metadata parsing never establishes that inference is supported.
    pub fn supports_inference(&self) -> bool {
        false
    }
}

/// Inspect the root `config.json` at `path` using the shared bounded reader.
pub fn inspect_root_config(path: &Path) -> Result<Inspection, InferenceError> {
    let text = read_config_json_bounded(path, "DeepSeek V4 root config")?;
    inspect_root_json(&text)
}

/// Inspect root JSON without loading weights or registering a model family.
pub fn inspect_root_json(text: &str) -> Result<Inspection, InferenceError> {
    if text.len() as u64 > MAX_CONFIG_JSON_BYTES {
        return Err(error("root config exceeds MAX_CONFIG_JSON_BYTES"));
    }
    let BoundedValue(value): BoundedValue = serde_json::from_str(text)
        .map_err(|reason| error(&format!("invalid root JSON: {reason}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| error("root must be an object"))?;
    let mut unsupported_fields: BTreeSet<String> = object
        .keys()
        .filter(|key| !VALIDATED_FIELDS.contains(&key.as_str()))
        .cloned()
        .collect();
    // These fields are preserved or checked structurally, but their execution semantics
    // require the separate architecture and tensor-layout work.
    unsupported_fields.extend(["compress_ratios".into(), "rope_scaling".into()]);
    let root: Root = serde_json::from_value(value)
        .map_err(|reason| error(&format!("invalid root fields: {reason}")))?;
    root.validate()?;
    Ok(Inspection {
        num_hidden_layers: root.num_hidden_layers,
        num_nextn_predict_layers: root.num_nextn_predict_layers,
        geometry: Geometry {
            hidden_size: root.hidden_size,
            num_attention_heads: root.num_attention_heads,
            head_dim: root.head_dim,
            qk_rope_head_dim: root.qk_rope_head_dim,
            num_key_value_heads: root.num_key_value_heads,
            vocab_size: root.vocab_size,
            max_position_embeddings: root.max_position_embeddings,
            n_routed_experts: root.n_routed_experts,
            n_shared_experts: root.n_shared_experts,
            num_experts_per_tok: root.num_experts_per_tok,
            moe_intermediate_size: root.moe_intermediate_size,
            index_head_dim: root.index_head_dim,
            o_groups: root.o_groups,
        },
        dspark_block_size: root.dspark_block_size,
        dspark_noise_token_id: root.dspark_noise_token_id,
        dspark_markov_rank: root.dspark_markov_rank,
        compress_ratios: root.compress_ratios,
        dspark_target_layer_ids: root.dspark_target_layer_ids,
        unsupported_fields: unsupported_fields.into_iter().collect(),
    })
}

fn error(message: &str) -> InferenceError {
    InferenceError::Inference(format!("DeepSeek V4 metadata: {message}"))
}

const VALIDATED_FIELDS: &[&str] = &[
    "architectures",
    "model_type",
    "num_hidden_layers",
    "num_nextn_predict_layers",
    "hidden_size",
    "num_attention_heads",
    "head_dim",
    "qk_rope_head_dim",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "num_hash_layers",
    "o_groups",
    "q_lora_rank",
    "o_lora_rank",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "sliding_window",
    "hc_mult",
    "hc_sinkhorn_iters",
    "dspark_block_size",
    "dspark_markov_rank",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "compress_ratios",
    "rms_norm_eps",
    "hc_eps",
    "rope_theta",
    "compress_rope_theta",
    "rope_scaling",
];

#[derive(Deserialize)]
struct Root {
    architectures: Vec<String>,
    model_type: String,
    num_hidden_layers: usize,
    num_nextn_predict_layers: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    num_hash_layers: usize,
    o_groups: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    index_head_dim: usize,
    index_n_heads: usize,
    index_topk: usize,
    sliding_window: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    dspark_block_size: usize,
    dspark_markov_rank: usize,
    dspark_noise_token_id: usize,
    dspark_target_layer_ids: Vec<usize>,
    compress_ratios: Vec<usize>,
    rms_norm_eps: f64,
    hc_eps: f64,
    rope_theta: f64,
    compress_rope_theta: f64,
    rope_scaling: RopeScaling,
}

#[derive(Deserialize)]
struct RopeScaling {
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    kind: String,
}

impl Root {
    fn validate(&self) -> Result<(), InferenceError> {
        if self.model_type != "deepseek_v4"
            || self.architectures.as_slice() != ["DeepseekV4ForCausalLM"]
        {
            return Err(error("unexpected model_type or architecture"));
        }
        for (name, value) in [
            ("num_hidden_layers", self.num_hidden_layers),
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("head_dim", self.head_dim),
            ("num_key_value_heads", self.num_key_value_heads),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("n_routed_experts", self.n_routed_experts),
            ("n_shared_experts", self.n_shared_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("o_groups", self.o_groups),
            ("q_lora_rank", self.q_lora_rank),
            ("o_lora_rank", self.o_lora_rank),
            ("index_head_dim", self.index_head_dim),
            ("index_n_heads", self.index_n_heads),
            ("index_topk", self.index_topk),
            ("sliding_window", self.sliding_window),
            ("hc_mult", self.hc_mult),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters),
            ("dspark_block_size", self.dspark_block_size),
            ("dspark_markov_rank", self.dspark_markov_rank),
            (
                "rope_scaling.original_max_position_embeddings",
                self.rope_scaling.original_max_position_embeddings,
            ),
        ] {
            if value == 0 {
                return Err(error(&format!("{name} must be positive")));
            }
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("hc_eps", self.hc_eps),
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
            ("rope_scaling.factor", self.rope_scaling.factor),
            ("rope_scaling.beta_fast", self.rope_scaling.beta_fast),
            ("rope_scaling.beta_slow", self.rope_scaling.beta_slow),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(error(&format!("{name} must be finite and positive")));
            }
        }
        if self.rope_scaling.kind.is_empty() {
            return Err(error("rope_scaling.type must not be empty"));
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return Err(error("num_experts_per_tok exceeds n_routed_experts"));
        }
        if self.num_hash_layers > self.num_hidden_layers {
            return Err(error("num_hash_layers exceeds base layers"));
        }
        if self.dspark_noise_token_id >= self.vocab_size {
            return Err(error("dspark_noise_token_id outside vocabulary"));
        }
        if self.dspark_target_layer_ids.is_empty() {
            return Err(error("dspark_target_layer_ids must not be empty"));
        }
        let mut seen = BTreeSet::new();
        for &id in &self.dspark_target_layer_ids {
            if id >= self.num_hidden_layers || !seen.insert(id) {
                return Err(error("dspark target layer IDs must be unique and in range"));
            }
        }
        if self.qk_rope_head_dim == 0
            || !self.qk_rope_head_dim.is_multiple_of(2)
            || self.qk_rope_head_dim > self.head_dim
        {
            return Err(error(
                "qk_rope_head_dim must be positive, even, and at most head_dim",
            ));
        }
        if self.compress_ratios.contains(&4) && self.qk_rope_head_dim > self.index_head_dim {
            return Err(error("qk_rope_head_dim exceeds index_head_dim for ratio 4"));
        }
        if self.compress_ratios.len() < self.num_hidden_layers {
            return Err(error("compression entries shorter than base layer count"));
        }
        let q_dim = self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| error("attention dimension overflow"))?;
        let group_dim = q_dim
            .checked_div(self.o_groups)
            .ok_or_else(|| error("invalid output group"))?;
        if group_dim == 0 {
            return Err(error("output group dimension is zero"));
        }
        q_dim
            .checked_mul(4)
            .ok_or_else(|| error("attention byte count overflow"))?;
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| error("key/value byte count overflow"))?;
        self.o_groups
            .checked_mul(self.o_lora_rank)
            .ok_or_else(|| error("output rank overflow"))?;
        self.hidden_size
            .checked_mul(self.dspark_target_layer_ids.len())
            .ok_or_else(|| error("target projection overflow"))?;
        self.hidden_size
            .checked_add(self.dspark_markov_rank)
            .ok_or_else(|| error("markov dimension overflow"))?;
        self.vocab_size
            .checked_mul(self.hidden_size)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| error("embedding byte count overflow"))?;
        self.n_routed_experts
            .checked_mul(self.moe_intermediate_size)
            .and_then(|n| n.checked_mul(self.hidden_size))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| error("expert byte count overflow"))?;
        if self.num_hidden_layers > MAX_ITEMS
            || [
                self.hidden_size,
                self.num_attention_heads,
                self.head_dim,
                self.num_key_value_heads,
                self.vocab_size,
                self.max_position_embeddings,
                self.n_routed_experts,
                self.n_shared_experts,
                self.moe_intermediate_size,
                self.o_groups,
                self.q_lora_rank,
                self.o_lora_rank,
                self.index_head_dim,
                self.index_n_heads,
                self.index_topk,
                self.sliding_window,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.dspark_block_size,
                self.dspark_markov_rank,
                self.num_nextn_predict_layers,
                self.rope_scaling.original_max_position_embeddings,
            ]
            .iter()
            .any(|&v| v > MAX_GEOMETRY)
        {
            return Err(error("metadata geometry exceeds inspection limit"));
        }
        Ok(())
    }
}

// A streaming JSON visitor prevents duplicate keys and caps arrays/objects before
// allocating elements, including inside fields whose semantics are unsupported.
struct BoundedValue(Value);

impl<'de> Deserialize<'de> for BoundedValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BoundedVisitor;
        impl<'de> Visitor<'de> for BoundedVisitor {
            type Value = BoundedValue;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("bounded JSON")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(BoundedValue(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(BoundedValue(Value::Number(v.into())))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(BoundedValue(Value::Number(v.into())))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Number::from_f64(v)
                    .map(|n| BoundedValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite number"))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(BoundedValue(Value::String(v.to_owned())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(BoundedValue(Value::String(v)))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(BoundedValue(Value::Null))
            }
            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(BoundedValue(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while values.len() < MAX_ITEMS {
                    match seq.next_element::<BoundedValue>()? {
                        Some(value) => values.push(value.0),
                        None => return Ok(BoundedValue(Value::Array(values))),
                    }
                }
                if seq.next_element::<IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("JSON array exceeds inspection limit"));
                }
                Ok(BoundedValue(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.len() == MAX_ITEMS {
                        return Err(de::Error::custom("JSON object exceeds inspection limit"));
                    }
                    if values.contains_key(&key) {
                        return Err(de::Error::custom(format!("duplicate key {key}")));
                    }
                    let value = map.next_value::<BoundedValue>()?;
                    values.insert(key, value.0);
                }
                Ok(BoundedValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(BoundedVisitor)
    }
}
