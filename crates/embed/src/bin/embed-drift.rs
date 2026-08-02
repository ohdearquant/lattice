//! Command-line embedding drift check and frozen-baseline updater.

#[cfg(not(target_arch = "wasm32"))]
mod cli {
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;
    use std::str::FromStr;

    use lattice_embed::EmbeddingModel;
    use lattice_embed::drift::{
        BaselineFixture, MAX_COSINE_DRIFT, ModelDriftOutcome, check_baseline, generate_baseline,
        load_baselines,
    };

    const USAGE: &str = "\
usage: embed-drift [--model NAME]... [--json] [--enforce] [--update-baseline] [-h|--help]

Compare current lattice embeddings with frozen release baselines.

options:
  --model <NAME>      Model to check. Repeat for multiple models.
                      With no --model, checks every model with a baseline fixture.
  --json              Emit one @@lattice {\"ev\":\"drift_done\",...} line to stdout.
  --enforce           Fail when a requested model's weights are absent.
                      LATTICE_DRIFT_GATE_ENFORCE=1 has the same effect.
  --update-baseline   Replace requested fixture files with vectors from this build.
                      Cannot be combined with --enforce.
  -h, --help          Print this help and exit.

exit codes:
  0  all completed comparisons are under threshold, or only visible non-enforced skips occurred
  1  drift detected: at least one checked model is over threshold
  2  enforced skip: at least one model's weights were absent while enforcing
  3  usage, fixture, model-loading, or IO error
";

    struct Options {
        models: Vec<EmbeddingModel>,
        emit_json: bool,
        enforce: bool,
        update_baseline: bool,
    }

    enum ParseAction {
        Run(Options),
        Help,
    }

    enum RowOutcome {
        Drift(ModelDriftOutcome),
        Updated,
    }

    struct ResultRow {
        model: String,
        outcome: RowOutcome,
    }

    fn usage(message: &str) -> ExitCode {
        eprintln!("ERROR: {message}\n");
        eprintln!("{USAGE}");
        ExitCode::from(3)
    }

    fn parse_args() -> std::result::Result<ParseAction, String> {
        let args = std::env::args().collect::<Vec<_>>();
        let mut models = Vec::new();
        let mut emit_json = false;
        let mut enforce =
            std::env::var("LATTICE_DRIFT_GATE_ENFORCE").is_ok_and(|value| value == "1");
        let mut update_baseline = false;
        let mut index = 1usize;

        while index < args.len() {
            match args[index].as_str() {
                "--model" => {
                    index += 1;
                    let name = args
                        .get(index)
                        .ok_or_else(|| "--model requires an argument".to_string())?;
                    let model = EmbeddingModel::from_str(name).map_err(|_| {
                        format!("--model '{name}' is not a recognised embedding model")
                    })?;
                    if !models.contains(&model) {
                        models.push(model);
                    }
                }
                "--json" => emit_json = true,
                "--enforce" => enforce = true,
                "--update-baseline" => update_baseline = true,
                "--help" | "-h" => return Ok(ParseAction::Help),
                other => return Err(format!("unknown argument: {other}")),
            }
            index += 1;
        }

        if update_baseline && enforce {
            return Err("--update-baseline cannot be combined with --enforce".to_string());
        }
        Ok(ParseAction::Run(Options {
            models,
            emit_json,
            enforce,
            update_baseline,
        }))
    }

    fn baseline_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("embed_drift_baseline_v1")
    }

    fn fixture_model(fixture: &BaselineFixture) -> std::result::Result<EmbeddingModel, String> {
        EmbeddingModel::from_str(&fixture.model).map_err(|error| {
            format!(
                "baseline fixture has invalid model name {:?}: {error}",
                fixture.model
            )
        })
    }

    fn fixture_for_model(
        fixtures: &[BaselineFixture],
        model: EmbeddingModel,
    ) -> std::result::Result<Option<&BaselineFixture>, String> {
        let mut found = None;
        for fixture in fixtures {
            if fixture_model(fixture)? == model {
                if found.is_some() {
                    return Err(format!("multiple baseline fixtures select model {model}"));
                }
                found = Some(fixture);
            }
        }
        Ok(found)
    }

    fn requested_models(
        options: &Options,
        fixtures: &[BaselineFixture],
    ) -> std::result::Result<Vec<EmbeddingModel>, String> {
        if !options.models.is_empty() {
            return Ok(options.models.clone());
        }
        if fixtures.is_empty() {
            return Err("baseline directory contains no JSON fixtures".to_string());
        }

        let mut models = Vec::with_capacity(fixtures.len());
        for fixture in fixtures {
            let model = fixture_model(fixture)?;
            if models.contains(&model) {
                return Err(format!("multiple baseline fixtures select model {model}"));
            }
            models.push(model);
        }
        Ok(models)
    }

    async fn run_checks(
        models: &[EmbeddingModel],
        fixtures: &[BaselineFixture],
    ) -> std::result::Result<Vec<ResultRow>, String> {
        let mut rows = Vec::with_capacity(models.len());
        for &model in models {
            let Some(fixture) = fixture_for_model(fixtures, model)? else {
                rows.push(ResultRow {
                    model: model.to_string(),
                    outcome: RowOutcome::Drift(ModelDriftOutcome::NoBaseline {
                        model: model.to_string(),
                    }),
                });
                continue;
            };
            let outcome = check_baseline(fixture)
                .await
                .map_err(|error| format!("drift check failed for {model}: {error}"))?;
            rows.push(ResultRow {
                model: model.to_string(),
                outcome: RowOutcome::Drift(outcome),
            });
        }
        Ok(rows)
    }

    async fn update_baselines(
        dir: &Path,
        models: &[EmbeddingModel],
        fixtures: &[BaselineFixture],
    ) -> std::result::Result<Vec<ResultRow>, String> {
        let template = fixtures
            .first()
            .ok_or_else(|| "cannot update baselines without an existing text corpus".to_string())?;
        let mut rows = Vec::with_capacity(models.len());
        for &model in models {
            let texts = fixture_for_model(fixtures, model)?
                .map(|fixture| fixture.texts.as_slice())
                .unwrap_or(template.texts.as_slice());
            let fixture = generate_baseline(model, texts)
                .await
                .map_err(|error| format!("failed to generate baseline for {model}: {error}"))?;
            let path = dir.join(fixture_filename(model));
            let bytes = serde_json::to_vec(&fixture)
                .map_err(|error| format!("failed to serialize baseline for {model}: {error}"))?;
            std::fs::write(&path, bytes).map_err(|error| {
                format!(
                    "failed to write baseline fixture {}: {error}",
                    path.display()
                )
            })?;
            rows.push(ResultRow {
                model: model.to_string(),
                outcome: RowOutcome::Updated,
            });
        }
        Ok(rows)
    }

    fn fixture_filename(model: EmbeddingModel) -> String {
        let stem = match model {
            EmbeddingModel::BgeSmallEnV15 => "bge_small_en_v15".to_string(),
            EmbeddingModel::MultilingualE5Small => "multilingual_e5_small".to_string(),
            EmbeddingModel::AllMiniLmL6V2 => "all_minilm_l6_v2".to_string(),
            EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
                "paraphrase_multilingual_minilm_l12_v2".to_string()
            }
            _ => model
                .to_string()
                .chars()
                .filter_map(|character| match character {
                    '-' => Some('_'),
                    '.' => None,
                    other => Some(other),
                })
                .collect(),
        };
        format!("{stem}.json")
    }

    fn print_rows(rows: &[ResultRow], enforce: bool) {
        eprintln!(
            "{:<48} {:<16} {:>13} {:>10} {:<13}",
            "model", "status", "max 1-cos", "threshold", "verdict"
        );
        for row in rows {
            match &row.outcome {
                RowOutcome::Drift(ModelDriftOutcome::Checked {
                    max_one_minus_cos, ..
                }) => {
                    let verdict = if *max_one_minus_cos < MAX_COSINE_DRIFT {
                        "PASS"
                    } else {
                        "DRIFT"
                    };
                    eprintln!(
                        "{:<48} {:<16} {:>13.6e} {:>10.1e} {:<13}",
                        row.model, "checked", max_one_minus_cos, MAX_COSINE_DRIFT, verdict
                    );
                }
                RowOutcome::Drift(ModelDriftOutcome::WeightsAbsent { .. }) => {
                    let verdict = if enforce { "ENFORCED_SKIP" } else { "SKIPPED" };
                    eprintln!(
                        "{:<48} {:<16} {:>13} {:>10.1e} {:<13}",
                        row.model, "weights_absent", "-", MAX_COSINE_DRIFT, verdict
                    );
                }
                RowOutcome::Drift(ModelDriftOutcome::NoBaseline { .. }) => {
                    eprintln!(
                        "{:<48} {:<16} {:>13} {:>10.1e} {:<13}",
                        row.model, "no_baseline", "-", MAX_COSINE_DRIFT, "ERROR"
                    );
                }
                RowOutcome::Updated => {
                    eprintln!(
                        "{:<48} {:<16} {:>13} {:>10.1e} {:<13}",
                        row.model, "updated", "-", MAX_COSINE_DRIFT, "UPDATED"
                    );
                }
            }
        }
    }

    fn emit_json(rows: &[ResultRow], enforce: bool) {
        let checked = rows
            .iter()
            .filter(|row| {
                matches!(
                    row.outcome,
                    RowOutcome::Drift(ModelDriftOutcome::Checked { .. })
                )
            })
            .count();
        let skipped = rows
            .iter()
            .filter(|row| {
                matches!(
                    row.outcome,
                    RowOutcome::Drift(ModelDriftOutcome::WeightsAbsent { .. })
                )
            })
            .count();
        let no_baseline = rows
            .iter()
            .filter(|row| {
                matches!(
                    row.outcome,
                    RowOutcome::Drift(ModelDriftOutcome::NoBaseline { .. })
                )
            })
            .count();
        let updated = rows
            .iter()
            .filter(|row| matches!(row.outcome, RowOutcome::Updated))
            .count();
        let results = rows
            .iter()
            .map(|row| match &row.outcome {
                RowOutcome::Drift(ModelDriftOutcome::Checked {
                    max_one_minus_cos,
                    worst_index,
                }) => serde_json::json!({
                    "model": row.model,
                    "status": "checked",
                    "max_one_minus_cos": max_one_minus_cos,
                    "worst_index": worst_index,
                    "threshold": MAX_COSINE_DRIFT,
                    "verdict": if *max_one_minus_cos < MAX_COSINE_DRIFT { "pass" } else { "drift" },
                }),
                RowOutcome::Drift(ModelDriftOutcome::WeightsAbsent { model }) => {
                    serde_json::json!({
                        "model": model,
                        "status": "weights_absent",
                        "max_one_minus_cos": null,
                        "worst_index": null,
                        "threshold": MAX_COSINE_DRIFT,
                        "verdict": if enforce { "enforced_skip" } else { "skipped" },
                    })
                }
                RowOutcome::Drift(ModelDriftOutcome::NoBaseline { model }) => {
                    serde_json::json!({
                        "model": model,
                        "status": "no_baseline",
                        "max_one_minus_cos": null,
                        "worst_index": null,
                        "threshold": MAX_COSINE_DRIFT,
                        "verdict": "error",
                    })
                }
                RowOutcome::Updated => serde_json::json!({
                    "model": row.model,
                    "status": "updated",
                    "max_one_minus_cos": null,
                    "worst_index": null,
                    "threshold": MAX_COSINE_DRIFT,
                    "verdict": "updated",
                }),
            })
            .collect::<Vec<_>>();
        let event = serde_json::json!({
            "ev": "drift_done",
            "results": results,
            "checked": checked,
            "skipped": skipped,
            "no_baseline": no_baseline,
            "updated": updated,
        });
        println!("@@lattice {event}");
    }

    fn check_exit_code(rows: &[ResultRow], enforce: bool) -> ExitCode {
        if rows.iter().any(|row| {
            matches!(
                row.outcome,
                RowOutcome::Drift(ModelDriftOutcome::NoBaseline { .. })
            )
        }) {
            return ExitCode::from(3);
        }
        if rows.iter().any(|row| {
            matches!(
                row.outcome,
                RowOutcome::Drift(ModelDriftOutcome::Checked {
                    max_one_minus_cos,
                    ..
                }) if max_one_minus_cos >= MAX_COSINE_DRIFT
            )
        }) {
            return ExitCode::from(1);
        }
        if enforce
            && rows.iter().any(|row| {
                matches!(
                    row.outcome,
                    RowOutcome::Drift(ModelDriftOutcome::WeightsAbsent { .. })
                )
            })
        {
            return ExitCode::from(2);
        }
        ExitCode::SUCCESS
    }

    #[tokio::main]
    pub(crate) async fn main() -> ExitCode {
        let options = match parse_args() {
            Ok(ParseAction::Help) => {
                eprintln!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            Ok(ParseAction::Run(options)) => options,
            Err(error) => return usage(&error),
        };
        let dir = baseline_dir();
        let fixtures = match load_baselines(&dir) {
            Ok(fixtures) => fixtures,
            Err(error) => return usage(&error.to_string()),
        };
        let models = match requested_models(&options, &fixtures) {
            Ok(models) => models,
            Err(error) => return usage(&error),
        };

        let rows = if options.update_baseline {
            match update_baselines(&dir, &models, &fixtures).await {
                Ok(rows) => rows,
                Err(error) => return usage(&error),
            }
        } else {
            match run_checks(&models, &fixtures).await {
                Ok(rows) => rows,
                Err(error) => return usage(&error),
            }
        };
        print_rows(&rows, options.enforce);
        if options.emit_json {
            emit_json(&rows, options.enforce);
        }
        if options.update_baseline {
            ExitCode::SUCCESS
        } else {
            check_exit_code(&rows, options.enforce)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    cli::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
