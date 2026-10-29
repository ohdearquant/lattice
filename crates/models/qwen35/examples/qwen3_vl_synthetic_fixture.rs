//! Offline JSON bridge for synthetic Qwen3-VL vision differential testing.
use lattice_model_qwen35::vision::checkpoint::{
    Qwen35VisionWeights, VisualBlockWeights, VisualMergerWeights,
};
use lattice_model_qwen35::vision::qwen3_vl_synthetic;
use serde_json::Value;
use std::io::{Error, ErrorKind, Read};

fn invalid() -> Error {
    Error::new(ErrorKind::InvalidInput, "invalid fixture JSON")
}
fn values(v: &Value, key: &str) -> Result<Vec<f32>, Error> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(invalid)?
        .iter()
        .map(|x| {
            let f = x.as_f64().ok_or_else(invalid)? as f32;
            if f.is_finite() { Ok(f) } else { Err(invalid()) }
        })
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::io::stdin().take(16_777_217).read_to_end(&mut bytes)?;
    if bytes.len() > 16_777_216 {
        return Err(invalid().into());
    }
    let v: Value = serde_json::from_slice(&bytes)?;
    let block = VisualBlockWeights {
        qkv_weight: values(&v, "qkv_weight")?,
        qkv_bias: values(&v, "qkv_bias")?,
        proj_weight: values(&v, "proj_weight")?,
        proj_bias: values(&v, "proj_bias")?,
        fc1_weight: values(&v, "fc1_weight")?,
        fc1_bias: values(&v, "fc1_bias")?,
        fc2_weight: values(&v, "fc2_weight")?,
        fc2_bias: values(&v, "fc2_bias")?,
        norm1_weight: values(&v, "norm1_weight")?,
        norm1_bias: values(&v, "norm1_bias")?,
        norm2_weight: values(&v, "norm2_weight")?,
        norm2_bias: values(&v, "norm2_bias")?,
    };
    let merger = VisualMergerWeights {
        fc1_weight: values(&v, "merger_fc1_weight")?,
        fc1_bias: values(&v, "merger_fc1_bias")?,
        fc2_weight: values(&v, "merger_fc2_weight")?,
        fc2_bias: values(&v, "merger_fc2_bias")?,
        norm_weight: values(&v, "merger_norm_weight")?,
        norm_bias: values(&v, "merger_norm_bias")?,
    };
    let weights = Qwen35VisionWeights {
        patch_embed_weight: values(&v, "patch_weight")?,
        patch_embed_weight_shape: vec![8, 3, 2, 16, 16],
        patch_embed_bias: values(&v, "patch_bias")?,
        pos_embed: values(&v, "positions")?,
        blocks: vec![block],
        merger,
    };
    let out = qwen3_vl_synthetic::run(&values(&v, "image")?, &weights)?;
    println!(
        "{}",
        serde_json::json!({"patches":out.patches, "patch_embedding":out.patch_embedding,
        "positions":out.positions, "block":out.block, "merged":out.merged})
    );
    Ok(())
}
