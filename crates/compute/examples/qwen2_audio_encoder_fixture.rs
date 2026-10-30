//! Synthetic JSON bridge for the Qwen2-Audio encoder differential.
use lattice_compute::audio::qwen2_audio::{encoder, layer, stem};
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
struct OwnedLayer {
    n1w: Vec<f32>,
    n1b: Vec<f32>,
    qw: Vec<f32>,
    qb: Vec<f32>,
    kw: Vec<f32>,
    vw: Vec<f32>,
    vb: Vec<f32>,
    ow: Vec<f32>,
    ob: Vec<f32>,
    n2w: Vec<f32>,
    n2b: Vec<f32>,
    f1w: Vec<f32>,
    f1b: Vec<f32>,
    f2w: Vec<f32>,
    f2b: Vec<f32>,
}
impl OwnedLayer {
    fn read(v: &Value) -> Result<Self, Error> {
        Ok(Self {
            n1w: values(v, "norm1_weight")?,
            n1b: values(v, "norm1_bias")?,
            qw: values(v, "q_weight")?,
            qb: values(v, "q_bias")?,
            kw: values(v, "k_weight")?,
            vw: values(v, "v_weight")?,
            vb: values(v, "v_bias")?,
            ow: values(v, "out_weight")?,
            ob: values(v, "out_bias")?,
            n2w: values(v, "norm2_weight")?,
            n2b: values(v, "norm2_bias")?,
            f1w: values(v, "fc1_weight")?,
            f1b: values(v, "fc1_bias")?,
            f2w: values(v, "fc2_weight")?,
            f2b: values(v, "fc2_bias")?,
        })
    }
    fn as_weights(&self) -> layer::LayerWeights<'_> {
        layer::LayerWeights {
            norm1_weight: &self.n1w,
            norm1_bias: &self.n1b,
            q_weight: &self.qw,
            q_bias: &self.qb,
            k_weight: &self.kw,
            v_weight: &self.vw,
            v_bias: &self.vb,
            out_weight: &self.ow,
            out_bias: &self.ob,
            norm2_weight: &self.n2w,
            norm2_bias: &self.n2b,
            fc1_weight: &self.f1w,
            fc1_bias: &self.f1b,
            fc2_weight: &self.f2w,
            fc2_bias: &self.f2b,
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::io::stdin().take(1_048_577).read_to_end(&mut bytes)?;
    if bytes.len() > 1_048_576 {
        return Err(invalid().into());
    }
    let v: Value = serde_json::from_slice(&bytes)?;
    let c = encoder::EncoderConfig {
        input_channels: dim(&v, "input_channels")?,
        frames: dim(&v, "frames")?,
        valid_frames: dim(&v, "valid_frames")?,
        hidden: dim(&v, "hidden")?,
        heads: dim(&v, "heads")?,
        ffn: dim(&v, "ffn")?,
        output: dim(&v, "output")?,
    };
    let input = values(&v, "input")?;
    let conv1 = values(&v, "conv1_weight")?;
    let bias1 = values(&v, "conv1_bias")?;
    let conv2 = values(&v, "conv2_weight")?;
    let bias2 = values(&v, "conv2_bias")?;
    let positions = values(&v, "positions")?;
    let nw = values(&v, "norm_weight")?;
    let nb = values(&v, "norm_bias")?;
    let pw = values(&v, "projector_weight")?;
    let pb = values(&v, "projector_bias")?;
    let owned: Vec<OwnedLayer> = v
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(invalid)?
        .iter()
        .map(OwnedLayer::read)
        .collect::<Result<_, _>>()?;
    let layers: Vec<_> = owned.iter().map(OwnedLayer::as_weights).collect();
    let w = encoder::EncoderWeights {
        stem: stem::StemWeights {
            conv1: &conv1,
            bias1: &bias1,
            conv2: &conv2,
            bias2: &bias2,
        },
        positions: &positions,
        layers: &layers,
        norm_weight: &nw,
        norm_bias: &nb,
        projector_weight: &pw,
        projector_bias: &pb,
    };
    let out = encoder::run(&input, &c, &w)?;
    println!(
        "{}",
        serde_json::json!({"stem":out.stem, "positioned":out.positioned, "layers":out.layers,
        "pooled":out.pooled, "normalized":out.normalized, "projected":out.projected,
        "time":out.time, "valid_tokens":out.valid_tokens})
    );
    Ok(())
}
