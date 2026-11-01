//! Synthetic JSON bridge for the Qwen2-Audio stem differential.
use lattice_compute::audio::qwen2_audio::stem::{StemWeights, run};
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
            let value = x.as_f64().ok_or_else(invalid)? as f32;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(invalid())
            }
        })
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = Vec::new();
    std::io::stdin().take(1_048_577).read_to_end(&mut input)?;
    if input.len() > 1_048_576 {
        return Err(invalid().into());
    }
    let v: Value = serde_json::from_slice(&input)?;
    let (input_channels, frames, hidden) = (
        dim(&v, "input_channels")?,
        dim(&v, "frames")?,
        dim(&v, "hidden")?,
    );
    let input = values(&v, "input")?;
    let conv1 = values(&v, "conv1")?;
    let bias1 = values(&v, "bias1")?;
    let conv2 = values(&v, "conv2")?;
    let bias2 = values(&v, "bias2")?;
    let weights = StemWeights {
        conv1: &conv1,
        bias1: &bias1,
        conv2: &conv2,
        bias2: &bias2,
    };
    let result = run(&input, input_channels, frames, hidden, &weights)?;
    println!(
        "{}",
        serde_json::json!({"first":result.first,"second":result.second,"output_frames":result.output_frames})
    );
    Ok(())
}
