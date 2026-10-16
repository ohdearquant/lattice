//! `lattice forecast`: point forecasts for a batch of univariate series, one JSON object per line.
//!
//! Input line: `{"id": <any, optional>, "target": [number | null, ...], "horizon": N,
//! "downsample": N (optional, default 1), "flip": bool (optional, default false)}`; `null` marks
//! a missing value. Output line, in input order: `{"id": <echoed>, "forecast": [number, ...]}`.
//! Any malformed line or failed forecast stops the run with its line number.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use lattice_inference::forecast::{PredictOptions, Reverso};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Model directory holding `args.json` and `model.safetensors`.
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

pub(crate) fn run(args: &Args) -> Result<(), String> {
    let model = Reverso::load(&args.model).map_err(|e| format!("{}: {e}", args.model.display()))?;
    let requests = read_requests(&args.input)?;
    let forecasts: Vec<Result<Vec<f32>, String>> = requests
        .par_iter()
        .enumerate()
        .map(|(i, r)| {
            let target: Vec<f32> = r
                .target
                .iter()
                .map(|v| v.map_or(f32::NAN, |x| x as f32))
                .collect();
            let options = PredictOptions {
                downsample: r.downsample.unwrap_or(1),
                flip_invariance: r.flip,
            };
            model
                .predict(&target, r.horizon, options)
                .map_err(|e| format!("request {}: {e}", i + 1))
        })
        .collect();

    let sink: Box<dyn Write> = if args.output == "-" {
        Box::new(std::io::stdout().lock())
    } else {
        Box::new(File::create(&args.output).map_err(|e| format!("{}: {e}", args.output))?)
    };
    let mut out = BufWriter::new(sink);
    for (request, forecast) in requests.iter().zip(forecasts) {
        let forecast = forecast?;
        let line = serde_json::json!({ "id": request.id, "forecast": forecast });
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
