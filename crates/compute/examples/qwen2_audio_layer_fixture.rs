//! Synthetic JSON bridge for the Qwen2-Audio encoder-layer differential.
use lattice_compute::audio::qwen2_audio::layer::{LayerWeights, run};
use serde_json::Value;
use std::io::{Error, ErrorKind, Read};

fn invalid() -> Error {
    Error::new(ErrorKind::InvalidInput, "invalid fixture JSON")
}
fn dim(v: &Value, key: &str) -> Result<usize, Error> {
    usize::try_from(v.get(key).and_then(Value::as_u64).ok_or_else(invalid)?).map_err(|_| invalid())
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
    std::io::stdin().take(1_048_577).read_to_end(&mut bytes)?;
    if bytes.len() > 1_048_576 {
        return Err(invalid().into());
    }
    let v: Value = serde_json::from_slice(&bytes)?;
    let (time, hidden, heads, ffn, valid) = (
        dim(&v, "time")?,
        dim(&v, "hidden")?,
        dim(&v, "heads")?,
        dim(&v, "ffn")?,
        dim(&v, "valid_keys")?,
    );
    let input = values(&v, "input")?;
    let n1w = values(&v, "norm1_weight")?;
    let n1b = values(&v, "norm1_bias")?;
    let qw = values(&v, "q_weight")?;
    let qb = values(&v, "q_bias")?;
    let kw = values(&v, "k_weight")?;
    let vw = values(&v, "v_weight")?;
    let vb = values(&v, "v_bias")?;
    let ow = values(&v, "out_weight")?;
    let ob = values(&v, "out_bias")?;
    let n2w = values(&v, "norm2_weight")?;
    let n2b = values(&v, "norm2_bias")?;
    let f1w = values(&v, "fc1_weight")?;
    let f1b = values(&v, "fc1_bias")?;
    let f2w = values(&v, "fc2_weight")?;
    let f2b = values(&v, "fc2_bias")?;
    let w = LayerWeights {
        norm1_weight: &n1w,
        norm1_bias: &n1b,
        q_weight: &qw,
        q_bias: &qb,
        k_weight: &kw,
        v_weight: &vw,
        v_bias: &vb,
        out_weight: &ow,
        out_bias: &ob,
        norm2_weight: &n2w,
        norm2_bias: &n2b,
        fc1_weight: &f1w,
        fc1_bias: &f1b,
        fc2_weight: &f2w,
        fc2_bias: &f2b,
    };
    let out = run(&input, time, hidden, heads, ffn, valid, &w)?;
    println!(
        "{}",
        serde_json::json!({"norm1":out.norm1,"query":out.query,"key":out.key,"value":out.value,"attention":out.attention,"residual":out.residual,"norm2":out.norm2,"activated":out.activated,"output":out.output})
    );
    Ok(())
}
