//! `lattice forecast`: forecasts for a batch of series, one JSON object per line.
//!
//! The model directory decides the model: `config.json` carrying a `chronos_config` loads
//! Chronos-2, one carrying a `layer_group_size` loads Toto 2.0, otherwise `args.json` loads Reverso.
//!
//! Input line: `{"id": <any, optional>, "target": [number | null, ...], "horizon": N,
//! "group": <any, optional>, "downsample": N (optional, default 1), "flip": bool (optional,
//! default false)}`; `null` marks a missing value. `downsample` and `flip` are Reverso options and
//! are refused for the others.
//!
//! Chronos-2 and Toto 2.0 also forecast series jointly. A `target` given as a list of lists is one
//! multivariate series whose variates attend to each other, and every line carrying the same
//! non-null `group` is forecast in one call in which all of their series attend to each other. A
//! joint call needs one horizon; shorter contexts are padded on the left with missing values.
//! Reverso refuses both.
//!
//! Output line, in input order: `{"id": <echoed>, "forecast": [number, ...]}`, where the forecast
//! is Reverso's point forecast or the quantile models' median; Chronos-2 and Toto 2.0 lines also
//! carry `"quantile_levels": [...]` and `"quantiles": [[...] per level]`. For a multivariate
//! target, `forecast` and `quantiles` hold one such entry per variate. Any malformed line or
//! failed forecast stops the run with its line number.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use lattice_inference::forecast::{
    Chronos2, Chronos2Series, PredictOptions, QuantileForecast, Reverso, Toto2, Toto2Variate,
};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Model directory: `config.json` (Chronos-2, Toto 2.0) or `args.json` (Reverso), and
    /// `model.safetensors`.
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Input JSON-lines file; `-` reads standard input.
    #[arg(long, default_value = "-")]
    pub(crate) input: String,
    /// Output JSON-lines file; `-` writes standard output.
    #[arg(long, default_value = "-")]
    pub(crate) output: String,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(untagged)]
enum Target {
    Univariate(Vec<Option<f64>>),
    Multivariate(Vec<Vec<Option<f64>>>),
}

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: Value,
    target: Target,
    horizon: usize,
    #[serde(default)]
    group: Value,
    #[serde(default)]
    downsample: Option<usize>,
    #[serde(default)]
    flip: bool,
}

impl Request {
    fn rows(&self) -> Vec<Vec<f32>> {
        let row = |r: &[Option<f64>]| r.iter().map(|v| v.map_or(f32::NAN, |x| x as f32)).collect();
        match &self.target {
            Target::Univariate(r) => vec![row(r)],
            Target::Multivariate(rs) => rs.iter().map(|r| row(r)).collect(),
        }
    }
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

/// Indices of requests forecast together: each ungrouped request alone, and all requests sharing a
/// `group` value together, in order of first appearance.
fn units(requests: &[Request]) -> Vec<Vec<usize>> {
    let mut units: Vec<Vec<usize>> = Vec::new();
    let mut by_group: HashMap<String, usize> = HashMap::new();
    for (i, r) in requests.iter().enumerate() {
        if r.group.is_null() {
            units.push(vec![i]);
            continue;
        }
        let key = r.group.to_string();
        match by_group.get(&key) {
            Some(&u) => units[u].push(i),
            None => {
                by_group.insert(key, units.len());
                units.push(vec![i]);
            }
        }
    }
    units
}

enum Model {
    Reverso(Reverso),
    Chronos2(Chronos2),
    Toto2(Toto2),
}

fn load_model(dir: &std::path::Path) -> Result<Model, String> {
    let err = |e: lattice_inference::InferenceError| format!("{}: {e}", dir.display());
    if let Ok(text) = std::fs::read_to_string(dir.join("config.json"))
        && let Ok(config) = serde_json::from_str::<Value>(&text)
    {
        if config.get("chronos_config").is_some() {
            return Chronos2::load(dir).map(Model::Chronos2).map_err(err);
        }
        if config.get("layer_group_size").is_some() {
            return Toto2::load(dir).map(Model::Toto2).map_err(err);
        }
    }
    Reverso::load(dir).map(Model::Reverso).map_err(err)
}

fn reverso_one(m: &Reverso, r: &Request) -> Result<Value, String> {
    let Target::Univariate(_) = &r.target else {
        return Err("Reverso forecasts one univariate series per line".into());
    };
    if !r.group.is_null() {
        return Err("Reverso does not forecast series jointly; remove `group`".into());
    }
    let options = PredictOptions {
        downsample: r.downsample.unwrap_or(1),
        flip_invariance: r.flip,
    };
    let forecast = m
        .predict(&r.rows()[0], r.horizon, options)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "id": r.id, "forecast": forecast }))
}

/// One joint call over every variate of the requests in `unit`, split back into one line each.
fn quantile_unit(model: &Model, requests: &[&Request]) -> Result<Vec<Value>, String> {
    let horizon = requests[0].horizon;
    if requests.iter().any(|r| r.horizon != horizon) {
        return Err("lines forecast jointly must share one horizon".into());
    }
    if requests
        .iter()
        .any(|r| r.downsample.is_some_and(|d| d != 1) || r.flip)
    {
        return Err("downsample and flip are Reverso options; this model takes neither".into());
    }
    let per_request: Vec<Vec<Vec<f32>>> = requests.iter().map(|r| r.rows()).collect();
    let longest = per_request
        .iter()
        .flatten()
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let rows: Vec<Vec<f32>> = per_request
        .iter()
        .flatten()
        .map(|r| {
            let mut padded = vec![f32::NAN; longest - r.len()];
            padded.extend_from_slice(r);
            padded
        })
        .collect();

    let (levels, outputs): (&[f32], Vec<QuantileForecast>) = match model {
        Model::Chronos2(m) => {
            let cfg = m.config();
            if horizon == 0 || horizon > cfg.max_horizon() {
                return Err(format!(
                    "horizon {horizon} is outside 1..={}",
                    cfg.max_horizon()
                ));
            }
            let series: Vec<Chronos2Series> = rows
                .into_iter()
                .map(|context| Chronos2Series {
                    context,
                    group_id: 0,
                    future_covariates: None,
                })
                .collect();
            let full = m
                .forward(&series, horizon.div_ceil(cfg.patch_size))
                .map_err(|e| e.to_string())?;
            let cut = full
                .iter()
                .map(|f| QuantileForecast {
                    horizon,
                    values: (0..cfg.quantiles.len())
                        .flat_map(|q| f.quantile(q)[..horizon].to_vec())
                        .collect(),
                })
                .collect();
            (&cfg.quantiles, cut)
        }
        Model::Toto2(m) => {
            let variates: Vec<Toto2Variate> = rows
                .into_iter()
                .map(|context| Toto2Variate {
                    context,
                    series_id: 0,
                })
                .collect();
            let out = m.forecast(&variates, horizon).map_err(|e| e.to_string())?;
            (m.quantile_levels(), out)
        }
        Model::Reverso(_) => return Err("Reverso does not forecast series jointly".into()),
    };

    let median = levels
        .iter()
        .position(|&q| q == 0.5)
        .ok_or("the model has no 0.5 quantile")?;
    let mut outputs = outputs.into_iter();
    let mut lines = Vec::with_capacity(requests.len());
    for r in requests {
        let line = match &r.target {
            Target::Univariate(_) => {
                let out = outputs.next().ok_or("missing forecast")?;
                let quantiles: Vec<&[f32]> = (0..levels.len()).map(|q| out.quantile(q)).collect();
                serde_json::json!({
                    "id": r.id,
                    "forecast": out.quantile(median),
                    "quantile_levels": levels,
                    "quantiles": quantiles,
                })
            }
            Target::Multivariate(rs) => {
                let outs: Vec<QuantileForecast> = outputs.by_ref().take(rs.len()).collect();
                if outs.len() != rs.len() {
                    return Err("missing forecast".into());
                }
                let medians: Vec<&[f32]> = outs.iter().map(|o| o.quantile(median)).collect();
                let quantiles: Vec<Vec<&[f32]>> = outs
                    .iter()
                    .map(|o| (0..levels.len()).map(|q| o.quantile(q)).collect())
                    .collect();
                serde_json::json!({
                    "id": r.id,
                    "forecast": medians,
                    "quantile_levels": levels,
                    "quantiles": quantiles,
                })
            }
        };
        lines.push(line);
    }
    Ok(lines)
}

fn forecast_unit(
    model: &Model,
    requests: &[Request],
    unit: &[usize],
) -> Vec<Result<Value, String>> {
    let fail = |e: String| {
        unit.iter()
            .map(|&i| Err(format!("request {}: {e}", i + 1)))
            .collect()
    };
    match model {
        Model::Reverso(m) => unit
            .iter()
            .map(|&i| reverso_one(m, &requests[i]).map_err(|e| format!("request {}: {e}", i + 1)))
            .collect(),
        _ => {
            let members: Vec<&Request> = unit.iter().map(|&i| &requests[i]).collect();
            match quantile_unit(model, &members) {
                Ok(lines) => lines.into_iter().map(Ok).collect(),
                Err(e) => fail(e),
            }
        }
    }
}

pub(crate) fn run(args: &Args) -> Result<(), String> {
    let model = load_model(&args.model)?;
    let requests = read_requests(&args.input)?;
    let units = units(&requests);
    let results: Vec<Vec<Result<Value, String>>> = units
        .par_iter()
        .map(|unit| forecast_unit(&model, &requests, unit))
        .collect();
    let mut forecasts: Vec<Option<Result<Value, String>>> =
        (0..requests.len()).map(|_| None).collect();
    for (unit, lines) in units.iter().zip(results) {
        for (&i, line) in unit.iter().zip(lines) {
            forecasts[i] = Some(line);
        }
    }

    let sink: Box<dyn Write> = if args.output == "-" {
        Box::new(std::io::stdout().lock())
    } else {
        Box::new(File::create(&args.output).map_err(|e| format!("{}: {e}", args.output))?)
    };
    let mut out = BufWriter::new(sink);
    for (i, line) in forecasts.into_iter().enumerate() {
        let line = line.ok_or_else(|| format!("request {}: no forecast produced", i + 1))??;
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
        assert_eq!(
            r.target,
            Target::Univariate(vec![Some(1.5), None, Some(3.0)])
        );
        assert_eq!((r.horizon, r.downsample, r.flip), (4, None, false));
        assert!(r.id.is_null() && r.group.is_null());
    }

    #[test]
    fn a_line_without_a_horizon_is_refused() {
        assert!(serde_json::from_str::<Request>(r#"{"target": [1.0]}"#).is_err());
    }

    #[test]
    fn a_list_of_lists_is_one_multivariate_series() {
        let r: Request =
            serde_json::from_str(r#"{"target": [[1, null], [2, 3]], "horizon": 1}"#).unwrap();
        assert_eq!(
            r.target,
            Target::Multivariate(vec![vec![Some(1.0), None], vec![Some(2.0), Some(3.0)]])
        );
        let rows = r.rows();
        assert_eq!(rows.len(), 2);
        assert!(rows[0][1].is_nan());
    }

    #[test]
    fn lines_sharing_a_group_form_one_unit_in_first_appearance_order() {
        let lines = [
            r#"{"target": [1], "horizon": 1, "group": "b"}"#,
            r#"{"target": [1], "horizon": 1}"#,
            r#"{"target": [1], "horizon": 1, "group": 7}"#,
            r#"{"target": [1], "horizon": 1, "group": "b"}"#,
            r#"{"target": [1], "horizon": 1, "group": "7"}"#,
        ];
        let requests: Vec<Request> = lines
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            units(&requests),
            vec![vec![0, 3], vec![1], vec![2], vec![4]]
        );
    }
}
