//! `lattice forecast`: forecasts for a batch of univariate series, one JSON object per line.
//!
//! The model directory decides the model: `config.json` carrying a `chronos_config` loads
//! Chronos-2, otherwise `args.json` loads Reverso.
//!
//! Input line: `{"id": <any, optional>, "target": [number | null, ...], "horizon": N,
//! "downsample": N (optional, default 1), "flip": bool (optional, default false)}`; `null` marks
//! a missing value. `downsample` and `flip` are Reverso options: Chronos-2 refuses a `downsample`
//! other than 1 and `flip: true`.
//! Output line, in input order: `{"id": <echoed>, "forecast": [number, ...]}`, where the forecast
//! is Reverso's point forecast or Chronos-2's median; Chronos-2 lines also carry
//! `"quantile_levels": [...]` and `"quantiles": [[...] per level]`. Any malformed line or failed
//! forecast stops the run with its line number.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use lattice_inference::forecast::{Chronos2, PredictOptions, Reverso};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Model directory: `config.json` (Chronos-2) or `args.json` (Reverso), and `model.safetensors`.
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Input JSON-lines file; `-` reads standard input.
    #[arg(long, default_value = "-")]
    pub(crate) input: String,
    /// Output JSON-lines file; `-` writes standard output.
    #[arg(long, default_value = "-")]
    pub(crate) output: String,
}

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: Value,
    target: Vec<Option<f64>>,
    horizon: usize,
    #[serde(default)]
    downsample: Option<usize>,
    #[serde(default)]
    flip: bool,
}

fn read_requests(input: &str) -> Result<Vec<Request>, String> {
    let reader: Box<dyn BufRead> = if input == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        let file = File::open(input).map_err(|e| format!("{input}: {e}"))?;
        Box::new(BufReader::new(file))
    };
    let mut requests = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {}: {e}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Request =
            serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", i + 1))?;
        requests.push(request);
    }
    Ok(requests)
}

enum Model {
    Reverso(Reverso),
    Chronos2(Chronos2),
}

fn load_model(dir: &std::path::Path) -> Result<Model, String> {
    let err = |e: lattice_inference::InferenceError| format!("{}: {e}", dir.display());
    if let Ok(text) = std::fs::read_to_string(dir.join("config.json"))
        && serde_json::from_str::<Value>(&text)
            .ok()
            .is_some_and(|v| v.get("chronos_config").is_some())
    {
        return Chronos2::load(dir).map(Model::Chronos2).map_err(err);
    }
    Reverso::load(dir).map(Model::Reverso).map_err(err)
}

fn forecast_one(model: &Model, r: &Request) -> Result<Value, String> {
    let target: Vec<f32> = r
        .target
        .iter()
        .map(|v| v.map_or(f32::NAN, |x| x as f32))
        .collect();
    match model {
        Model::Reverso(m) => {
            let options = PredictOptions {
                downsample: r.downsample.unwrap_or(1),
                flip_invariance: r.flip,
            };
            let forecast = m
                .predict(&target, r.horizon, options)
                .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({ "id": r.id, "forecast": forecast }))
        }
        Model::Chronos2(m) => {
            if r.downsample.is_some_and(|d| d != 1) || r.flip {
                return Err(
                    "downsample and flip are Reverso options; Chronos-2 takes neither".into(),
                );
            }
            let out = m.predict(&target, r.horizon).map_err(|e| e.to_string())?;
            let levels = &m.config().quantiles;
            let median = levels
                .iter()
                .position(|&q| q == 0.5)
                .ok_or("the model has no 0.5 quantile")?;
            let quantiles: Vec<&[f32]> = (0..levels.len()).map(|q| out.quantile(q)).collect();
            Ok(serde_json::json!({
                "id": r.id,
                "forecast": out.quantile(median),
                "quantile_levels": levels,
                "quantiles": quantiles,
            }))
        }
    }
}

pub(crate) fn run(args: &Args) -> Result<(), String> {
    let model = load_model(&args.model)?;
    let requests = read_requests(&args.input)?;
    let forecasts: Vec<Result<Value, String>> = requests
        .par_iter()
        .enumerate()
        .map(|(i, r)| forecast_one(&model, r).map_err(|e| format!("request {}: {e}", i + 1)))
        .collect();

    let sink: Box<dyn Write> = if args.output == "-" {
        Box::new(std::io::stdout().lock())
    } else {
        Box::new(File::create(&args.output).map_err(|e| format!("{}: {e}", args.output))?)
    };
    let mut out = BufWriter::new(sink);
    for line in forecasts {
        let line = line?;
        writeln!(out, "{line}").map_err(|e| format!("write: {e}"))?;
    }
    out.flush().map_err(|e| format!("write: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_line_parses_missing_values_and_defaults() {
        let r: Request =
            serde_json::from_str(r#"{"target": [1.5, null, 3], "horizon": 4}"#).unwrap();
        assert_eq!(r.target, vec![Some(1.5), None, Some(3.0)]);
        assert_eq!((r.horizon, r.downsample, r.flip), (4, None, false));
        assert!(r.id.is_null());
    }

    #[test]
    fn a_line_without_a_horizon_is_refused() {
        assert!(serde_json::from_str::<Request>(r#"{"target": [1.0]}"#).is_err());
    }
}
