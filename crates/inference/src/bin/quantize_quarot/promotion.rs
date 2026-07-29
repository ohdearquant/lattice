use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use lattice_inference::quant::q4_manifest::{
    ManifestFlavor, load_manifest, read_manifest_bytes_bounded, verify_q4_source_provenance,
};

pub const ACCEPTANCE_RECEIPT_FILE: &str = "quarot_ppl_acceptance.json";

static NEXT_STAGE_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct PplGateConfig {
    pub evaluator: PathBuf,
    pub source_model_dir: PathBuf,
    pub baseline_q4_dir: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub corpus_file: PathBuf,
    pub window: usize,
    pub stride: usize,
    pub max_tokens: Option<usize>,
    pub delta_threshold: f64,
    pub rotation_seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PplMeasurement {
    pub label: String,
    pub ppl: f64,
    pub nll: f64,
    pub tokens: u64,
    pub windows: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PplGateEvidence {
    pub unrotated: PplMeasurement,
    pub quarot: PplMeasurement,
    pub delta: f64,
    pub threshold: f64,
}

#[derive(Debug, Serialize)]
struct AcceptanceReceipt {
    schema_version: u32,
    gate: &'static str,
    accepted: bool,
    artifact_manifest_sha256: String,
    evaluator: String,
    evaluator_sha256: String,
    baseline_q4_dir: String,
    baseline_manifest_sha256: String,
    tokenizer_dir: String,
    tokenizer_sha256: String,
    corpus_file: String,
    corpus_sha256: String,
    window: usize,
    stride: usize,
    max_tokens: Option<usize>,
    rotation_seed: u64,
    evidence: PplGateEvidence,
}

pub fn run_ppl_evaluator(
    config: &PplGateConfig,
    quarot_q4_dir: &Path,
) -> Result<PplGateEvidence, String> {
    validate_gate_config(config)?;

    let mut command = Command::new(&config.evaluator);
    command
        .arg("--q4-dir")
        .arg(&config.baseline_q4_dir)
        .arg("--quarot-q4-dir")
        .arg(quarot_q4_dir)
        .arg("--tokenizer-dir")
        .arg(&config.tokenizer_dir)
        .arg("--corpus-file")
        .arg(&config.corpus_file)
        .arg("--window")
        .arg(config.window.to_string())
        .arg("--stride")
        .arg(config.stride.to_string())
        .arg("--delta-threshold")
        .arg(config.delta_threshold.to_string())
        .arg("--json");
    if let Some(max_tokens) = config.max_tokens {
        command.arg("--max-tokens").arg(max_tokens.to_string());
    }

    let output = command.output().map_err(|error| {
        format!(
            "failed to execute explicit PPL evaluator {}: {error}",
            config.evaluator.display()
        )
    })?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("PPL evaluator stdout was not UTF-8: {error}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");

    if !output.status.success() {
        return Err(format!(
            "PPL acceptance evaluator exited with {}; converted artifact was not promoted",
            output.status
        ));
    }

    parse_ppl_evidence(&stdout, config.delta_threshold)
}

pub fn promote_with_gate<T, C, E>(
    output_dir: &Path,
    dry_run: bool,
    gate_config: &PplGateConfig,
    convert: C,
    evaluate: E,
) -> Result<(T, PplGateEvidence), String>
where
    C: FnOnce(&Path) -> Result<T, String>,
    E: FnOnce(&PplGateConfig, &Path) -> Result<PplGateEvidence, String>,
{
    validate_gate_config(gate_config)?;
    if !dry_run {
        validate_destination(output_dir)?;
    }

    let mut stage = StagingDir::new(output_dir)?;
    let conversion = convert(stage.path())?;
    let evidence = evaluate(gate_config, stage.path())?;
    validate_gate_evidence(&evidence, gate_config.delta_threshold)?;
    write_receipt(stage.path(), gate_config, &evidence)?;

    if dry_run {
        return Ok((conversion, evidence));
    }

    remove_empty_destination(output_dir)?;
    fs::rename(stage.path(), output_dir).map_err(|error| {
        format!(
            "failed to promote accepted QuaRot artifact {} -> {}: {error}",
            stage.path().display(),
            output_dir.display()
        )
    })?;
    stage.disarm();
    Ok((conversion, evidence))
}

fn validate_gate_config(config: &PplGateConfig) -> Result<(), String> {
    if !config.evaluator.is_absolute() {
        return Err(format!(
            "--ppl-evaluator must be an absolute path, got {}",
            config.evaluator.display()
        ));
    }
    let metadata = fs::metadata(&config.evaluator).map_err(|error| {
        format!(
            "cannot stat explicit PPL evaluator {}: {error}",
            config.evaluator.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "explicit PPL evaluator is not a file: {}",
            config.evaluator.display()
        ));
    }
    if !config.baseline_q4_dir.is_dir() {
        return Err(format!(
            "--baseline-q4-dir is not a directory: {}",
            config.baseline_q4_dir.display()
        ));
    }
    let baseline_manifest = load_manifest(&config.baseline_q4_dir)
        .map_err(|error| format!("invalid --baseline-q4-dir manifest: {error}"))?
        .ok_or_else(|| {
            format!(
                "--baseline-q4-dir has no quantize_index.json: {}",
                config.baseline_q4_dir.display()
            )
        })?;
    if baseline_manifest.flavor != ManifestFlavor::QuantizeQ4 {
        return Err(format!(
            "--baseline-q4-dir must be an unrotated quantize_q4 artifact: {}",
            config.baseline_q4_dir.display()
        ));
    }
    verify_q4_source_provenance(&config.source_model_dir, &config.baseline_q4_dir)?;
    let tokenizer_path = config.tokenizer_dir.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        return Err(format!(
            "--tokenizer-dir has no tokenizer.json: {}",
            config.tokenizer_dir.display()
        ));
    }
    if !config.corpus_file.is_file() {
        return Err(format!(
            "--corpus-file is not a file: {}",
            config.corpus_file.display()
        ));
    }
    if !config.delta_threshold.is_finite() {
        return Err("--delta-threshold must be finite".into());
    }
    if config.window < 2 {
        return Err("--window must be at least 2".into());
    }
    if config.stride == 0 || config.stride >= config.window {
        return Err("--stride must be greater than 0 and less than --window".into());
    }
    if config.max_tokens.is_some_and(|value| value < 2) {
        return Err("--max-tokens must be at least 2".into());
    }
    Ok(())
}

fn parse_ppl_evidence(stdout: &str, threshold: f64) -> Result<PplGateEvidence, String> {
    if !threshold.is_finite() {
        return Err("PPL acceptance threshold must be finite".into());
    }

    let mut unrotated = None;
    let mut quarot = None;
    for line in stdout.lines() {
        let Some(payload) = line.strip_prefix("@@lattice ") else {
            continue;
        };
        let value: serde_json::Value = serde_json::from_str(payload)
            .map_err(|error| format!("malformed structured evaluator event: {error}"))?;
        if value.get("ev").and_then(serde_json::Value::as_str) != Some("perplexity") {
            continue;
        }
        let measurement = parse_measurement(&value)?;
        match measurement.label.as_str() {
            "q4" => {
                if unrotated.replace(measurement).is_some() {
                    return Err("PPL evaluator emitted duplicate q4 measurements".into());
                }
            }
            "quarot" => {
                if quarot.replace(measurement).is_some() {
                    return Err("PPL evaluator emitted duplicate quarot measurements".into());
                }
            }
            _ => {}
        }
    }

    let unrotated =
        unrotated.ok_or_else(|| "PPL evaluator emitted no q4 measurement".to_string())?;
    let quarot = quarot.ok_or_else(|| "PPL evaluator emitted no quarot measurement".to_string())?;
    if unrotated.tokens != quarot.tokens || unrotated.windows != quarot.windows {
        return Err(format!(
            "PPL measurements used different coverage: q4={} tokens/{} windows, quarot={} tokens/{} windows",
            unrotated.tokens, unrotated.windows, quarot.tokens, quarot.windows
        ));
    }

    let delta = quarot.ppl - unrotated.ppl;
    if !delta.is_finite() {
        return Err("PPL evaluator produced a non-finite delta".into());
    }
    if delta >= threshold {
        return Err(format!(
            "PPL evaluator exited 0 but evidence fails acceptance: delta {delta:.6} >= threshold {threshold:.6}"
        ));
    }

    let evidence = PplGateEvidence {
        unrotated,
        quarot,
        delta,
        threshold,
    };
    validate_gate_evidence(&evidence, threshold)?;
    Ok(evidence)
}

fn validate_gate_evidence(evidence: &PplGateEvidence, threshold: f64) -> Result<(), String> {
    if evidence.threshold.to_bits() != threshold.to_bits() {
        return Err("PPL evidence threshold does not match the requested policy".into());
    }
    if evidence.unrotated.label != "q4" || evidence.quarot.label != "quarot" {
        return Err("PPL evidence labels must be q4 and quarot".into());
    }
    if evidence.unrotated.tokens == 0
        || evidence.unrotated.windows == 0
        || evidence.unrotated.tokens != evidence.quarot.tokens
        || evidence.unrotated.windows != evidence.quarot.windows
    {
        return Err("PPL evidence must have positive, matching token/window coverage".into());
    }
    let recomputed = evidence.quarot.ppl - evidence.unrotated.ppl;
    if !evidence.unrotated.ppl.is_finite()
        || evidence.unrotated.ppl <= 0.0
        || !evidence.quarot.ppl.is_finite()
        || evidence.quarot.ppl <= 0.0
        || !evidence.unrotated.nll.is_finite()
        || !evidence.quarot.nll.is_finite()
        || !recomputed.is_finite()
        || recomputed.to_bits() != evidence.delta.to_bits()
    {
        return Err("PPL evidence contains invalid or internally inconsistent numerics".into());
    }
    if evidence.delta >= threshold {
        return Err(format!(
            "PPL evidence fails acceptance: delta {:.6} >= threshold {:.6}",
            evidence.delta, threshold
        ));
    }
    Ok(())
}

fn parse_measurement(value: &serde_json::Value) -> Result<PplMeasurement, String> {
    let label = value
        .get("label")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "perplexity event is missing a string label".to_string())?
        .to_string();
    let ppl = finite_positive_number(value, "ppl", &label)?;
    let nll = finite_number(value, "nll", &label)?;
    let tokens = positive_integer(value, "tokens", &label)?;
    let windows = positive_integer(value, "windows", &label)?;
    Ok(PplMeasurement {
        label,
        ppl,
        nll,
        tokens,
        windows,
    })
}

fn finite_positive_number(
    value: &serde_json::Value,
    field: &str,
    label: &str,
) -> Result<f64, String> {
    let number = finite_number(value, field, label)?;
    if number <= 0.0 {
        return Err(format!(
            "perplexity event {label:?} field {field:?} must be positive"
        ));
    }
    Ok(number)
}

fn finite_number(value: &serde_json::Value, field: &str, label: &str) -> Result<f64, String> {
    let number = value
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("perplexity event {label:?} is missing numeric field {field:?}"))?;
    if !number.is_finite() {
        return Err(format!(
            "perplexity event {label:?} field {field:?} must be finite"
        ));
    }
    Ok(number)
}

fn positive_integer(value: &serde_json::Value, field: &str, label: &str) -> Result<u64, String> {
    let number = value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("perplexity event {label:?} is missing integer field {field:?}"))?;
    if number == 0 {
        return Err(format!(
            "perplexity event {label:?} field {field:?} must be positive"
        ));
    }
    Ok(number)
}

fn validate_destination(output_dir: &Path) -> Result<(), String> {
    let metadata = match fs::symlink_metadata(output_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "cannot inspect output directory {}: {error}",
                output_dir.display()
            ));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "output path must be an absent or empty real directory: {}",
            output_dir.display()
        ));
    }
    let mut entries = fs::read_dir(output_dir).map_err(|error| {
        format!(
            "cannot read output directory {}: {error}",
            output_dir.display()
        )
    })?;
    if entries.next().is_some() {
        return Err(format!(
            "output directory is not empty; refusing staged promotion: {}",
            output_dir.display()
        ));
    }
    Ok(())
}

fn remove_empty_destination(output_dir: &Path) -> Result<(), String> {
    match fs::symlink_metadata(output_dir) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "cannot inspect output directory {} before promotion: {error}",
                output_dir.display()
            ));
        }
    }
    validate_destination(output_dir)?;
    fs::remove_dir(output_dir).map_err(|error| {
        format!(
            "failed to remove empty output placeholder {} before promotion: {error}",
            output_dir.display()
        )
    })
}

fn write_receipt(
    stage_dir: &Path,
    config: &PplGateConfig,
    evidence: &PplGateEvidence,
) -> Result<(), String> {
    let manifest = stage_dir.join("quantize_index.json");
    let parsed_manifest = load_manifest(stage_dir)
        .map_err(|error| format!("accepted artifact manifest is invalid: {error}"))?
        .ok_or_else(|| format!("accepted artifact has no manifest: {}", manifest.display()))?;
    if parsed_manifest.flavor != ManifestFlavor::QuaRot
        || parsed_manifest.quarot_seed != Some(config.rotation_seed)
    {
        return Err(format!(
            "accepted artifact manifest does not record requested QuaRot seed {}",
            config.rotation_seed
        ));
    }
    let manifest_bytes = read_manifest_bytes_bounded(&manifest)
        .map_err(|error| format!("accepted artifact manifest is unreadable: {error}"))?
        .ok_or_else(|| format!("accepted artifact has no manifest: {}", manifest.display()))?;
    let receipt = AcceptanceReceipt {
        schema_version: 1,
        gate: "adr-044-quarot-ppl-delta",
        accepted: true,
        artifact_manifest_sha256: format!("{:x}", Sha256::digest(&manifest_bytes)),
        evaluator: config.evaluator.display().to_string(),
        evaluator_sha256: sha256_file(&config.evaluator)?,
        baseline_q4_dir: config.baseline_q4_dir.display().to_string(),
        baseline_manifest_sha256: sha256_file(&config.baseline_q4_dir.join("quantize_index.json"))?,
        tokenizer_dir: config.tokenizer_dir.display().to_string(),
        tokenizer_sha256: sha256_file(&config.tokenizer_dir.join("tokenizer.json"))?,
        corpus_file: config.corpus_file.display().to_string(),
        corpus_sha256: sha256_file(&config.corpus_file)?,
        window: config.window,
        stride: config.stride,
        max_tokens: config.max_tokens,
        rotation_seed: config.rotation_seed,
        evidence: evidence.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("failed to serialize PPL acceptance receipt: {error}"))?;
    fs::write(stage_dir.join(ACCEPTANCE_RECEIPT_FILE), bytes)
        .map_err(|error| format!("failed to write PPL acceptance receipt: {error}"))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("failed to hash gate input {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash gate input {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

struct StagingDir {
    path: PathBuf,
    armed: bool,
}

impl StagingDir {
    fn new(output_dir: &Path) -> Result<Self, String> {
        let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
        let name = output_dir
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                format!(
                    "output directory must end in a valid UTF-8 name: {}",
                    output_dir.display()
                )
            })?;
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create output parent {}: {error}",
                parent.display()
            )
        })?;
        for _ in 0..128 {
            let id = NEXT_STAGE_ID.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".{name}.quarot-staging-{}-{id}",
                std::process::id()
            ));
            if !path.exists() {
                return Ok(Self { path, armed: true });
            }
        }
        Err(format!(
            "could not allocate a unique staging directory beside {}",
            output_dir.display()
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if self.armed && self.path.exists() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn gate_config(root: &Path) -> PplGateConfig {
        fs::create_dir_all(root.join("baseline")).unwrap();
        fs::create_dir_all(root.join("model")).unwrap();
        fs::create_dir_all(root.join("tokenizer")).unwrap();
        fs::write(root.join("eval_perplexity"), b"stub evaluator").unwrap();
        fs::write(root.join("baseline/quantize_index.json"), b"[]").unwrap();
        fs::write(root.join("model/config.json"), b"{}").unwrap();
        fs::write(root.join("model/model.safetensors"), b"source checkpoint").unwrap();
        lattice_inference::quant::q4_manifest::write_q4_source_provenance(
            &root.join("model"),
            &root.join("baseline"),
        )
        .unwrap();
        fs::write(root.join("tokenizer/tokenizer.json"), b"{}").unwrap();
        fs::write(root.join("corpus.txt"), b"held out corpus").unwrap();
        PplGateConfig {
            evaluator: root.join("eval_perplexity"),
            source_model_dir: root.join("model"),
            baseline_q4_dir: root.join("baseline"),
            tokenizer_dir: root.join("tokenizer"),
            corpus_file: root.join("corpus.txt"),
            window: 16,
            stride: 8,
            max_tokens: Some(32),
            delta_threshold: 0.5,
            rotation_seed: 1,
        }
    }

    fn passing_evidence() -> PplGateEvidence {
        PplGateEvidence {
            unrotated: PplMeasurement {
                label: "q4".into(),
                ppl: 10.0,
                nll: 2.3,
                tokens: 31,
                windows: 3,
            },
            quarot: PplMeasurement {
                label: "quarot".into(),
                ppl: 10.25,
                nll: 2.32,
                tokens: 31,
                windows: 3,
            },
            delta: 0.25,
            threshold: 0.5,
        }
    }

    #[test]
    fn promotion_requires_gate_evidence_before_publishing_output() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("accepted");
        let calls = RefCell::new(Vec::new());
        let config = gate_config(temp.path());

        let result = promote_with_gate(
            &output,
            false,
            &config,
            |stage| {
                calls.borrow_mut().push("convert");
                fs::create_dir_all(stage).unwrap();
                fs::write(
                    stage.join("quantize_index.json"),
                    b"{\"quarot_seed\":1,\"tensors\":[]}",
                )
                .unwrap();
                Ok(7_u32)
            },
            |_, stage| {
                calls.borrow_mut().push("gate");
                assert!(stage.join("quantize_index.json").is_file());
                assert!(!output.exists());
                Ok(passing_evidence())
            },
        )
        .unwrap();

        assert_eq!(result.0, 7);
        assert_eq!(&*calls.borrow(), &["convert", "gate"]);
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join(ACCEPTANCE_RECEIPT_FILE)).unwrap())
                .unwrap();
        assert_eq!(receipt["accepted"], true);
        assert_eq!(receipt["rotation_seed"], 1);
        assert_eq!(receipt["evidence"]["delta"], 0.25);
        assert_eq!(
            receipt["artifact_manifest_sha256"].as_str().unwrap().len(),
            64
        );
    }

    #[test]
    fn gate_failure_removes_staging_and_never_publishes_output() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("rejected");
        let config = gate_config(temp.path());

        let error = promote_with_gate(
            &output,
            false,
            &config,
            |stage| {
                fs::create_dir_all(stage).unwrap();
                fs::write(
                    stage.join("quantize_index.json"),
                    b"{\"quarot_seed\":1,\"tensors\":[]}",
                )
                .unwrap();
                Ok(())
            },
            |_, _| Err("stub PPL rejection".into()),
        )
        .unwrap_err();

        assert_eq!(error, "stub PPL rejection");
        assert!(!output.exists());
        assert!(fs::read_dir(temp.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("staging")
        }));
    }

    #[test]
    fn evaluator_exit_zero_without_structured_evidence_fails_closed() {
        let error = parse_ppl_evidence("Verdict: PASS\n", 0.5).unwrap_err();
        assert_eq!(error, "PPL evaluator emitted no q4 measurement");
    }

    #[test]
    fn evaluator_exit_zero_with_failing_delta_fails_closed() {
        let stdout = concat!(
            "@@lattice {\"ev\":\"perplexity\",\"label\":\"q4\",\"ppl\":10.0,",
            "\"nll\":2.3,\"tokens\":31,\"windows\":3,\"ms\":1}\n",
            "@@lattice {\"ev\":\"perplexity\",\"label\":\"quarot\",\"ppl\":10.5,",
            "\"nll\":2.35,\"tokens\":31,\"windows\":3,\"ms\":1}\n"
        );
        let error = parse_ppl_evidence(stdout, 0.5).unwrap_err();
        assert!(error.contains("delta 0.500000 >= threshold 0.500000"));
    }

    #[test]
    fn rotated_artifact_cannot_be_used_as_the_unrotated_baseline() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("never-converted");
        let config = gate_config(temp.path());
        fs::write(
            config.baseline_q4_dir.join("quantize_index.json"),
            b"{\"quarot_seed\":1,\"tensors\":[]}",
        )
        .unwrap();
        let converted = RefCell::new(false);

        let error = promote_with_gate(
            &output,
            false,
            &config,
            |_| {
                *converted.borrow_mut() = true;
                Ok(())
            },
            |_, _| Ok(passing_evidence()),
        )
        .unwrap_err();

        assert!(error.contains("must be an unrotated quantize_q4 artifact"));
        assert!(!*converted.borrow());
        assert!(!output.exists());
    }

    #[test]
    fn valid_q4_baseline_from_different_source_is_rejected_before_conversion() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("never-converted");
        let config = gate_config(temp.path());
        let other_source = temp.path().join("other-model");
        fs::create_dir_all(&other_source).unwrap();
        fs::write(other_source.join("config.json"), b"{}").unwrap();
        fs::write(
            other_source.join("model.safetensors"),
            b"different source checkpoint",
        )
        .unwrap();
        lattice_inference::quant::q4_manifest::write_q4_source_provenance(
            &other_source,
            &config.baseline_q4_dir,
        )
        .unwrap();
        let converted = RefCell::new(false);

        let error = promote_with_gate(
            &output,
            false,
            &config,
            |_| {
                *converted.borrow_mut() = true;
                Ok(())
            },
            |_, _| Ok(passing_evidence()),
        )
        .unwrap_err();

        assert!(error.contains("was produced from a different source checkpoint"));
        assert!(!*converted.borrow());
        assert!(!output.exists());
    }

    #[test]
    fn artifact_seed_mismatch_cannot_receive_acceptance_receipt() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("wrong-seed");
        let config = gate_config(temp.path());

        let error = promote_with_gate(
            &output,
            false,
            &config,
            |stage| {
                fs::create_dir_all(stage).unwrap();
                fs::write(
                    stage.join("quantize_index.json"),
                    b"{\"quarot_seed\":2,\"tensors\":[]}",
                )
                .unwrap();
                Ok(())
            },
            |_, _| Ok(passing_evidence()),
        )
        .unwrap_err();

        assert!(error.contains("does not record requested QuaRot seed 1"));
        assert!(!output.exists());
    }

    #[cfg(unix)]
    #[test]
    fn explicit_stub_evaluator_is_executed_and_recorded() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let evaluator = temp.path().join("stub-eval");
        fs::write(
            &evaluator,
            concat!(
                "#!/bin/sh\n",
                "test \"$#\" -eq 17 || exit 9\n",
                "test \"$1\" = \"--q4-dir\" || exit 9\n",
                "test \"$3\" = \"--quarot-q4-dir\" || exit 9\n",
                "test \"$5\" = \"--tokenizer-dir\" || exit 9\n",
                "test \"$7\" = \"--corpus-file\" || exit 9\n",
                "test \"$9\" = \"--window\" && test \"${10}\" = \"16\" || exit 9\n",
                "test \"${11}\" = \"--stride\" && test \"${12}\" = \"8\" || exit 9\n",
                "test \"${13}\" = \"--delta-threshold\" || exit 9\n",
                "test \"${14}\" = \"0.5\" && test \"${15}\" = \"--json\" || exit 9\n",
                "test \"${16}\" = \"--max-tokens\" && test \"${17}\" = \"32\" || exit 9\n",
                "printf '%s\\n' ",
                "'@@lattice {\"ev\":\"perplexity\",\"label\":\"q4\",\"ppl\":10.0,",
                "\"nll\":2.3,\"tokens\":31,\"windows\":3,\"ms\":1}'\n",
                "printf '%s\\n' ",
                "'@@lattice {\"ev\":\"perplexity\",\"label\":\"quarot\",\"ppl\":10.2,",
                "\"nll\":2.32,\"tokens\":31,\"windows\":3,\"ms\":1}'\n",
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&evaluator).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&evaluator, permissions).unwrap();
        let mut config = gate_config(temp.path());
        config.evaluator = evaluator;

        let evidence = run_ppl_evaluator(&config, &temp.path().join("staged")).unwrap();

        assert!((evidence.delta - 0.2).abs() < 1e-12);
        assert_eq!(evidence.threshold, 0.5);
    }

    #[test]
    fn dry_run_executes_gate_but_does_not_publish_output() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("dry-run-output");
        let config = gate_config(temp.path());
        let called = RefCell::new(false);

        promote_with_gate(
            &output,
            true,
            &config,
            |stage| {
                fs::create_dir_all(stage).unwrap();
                fs::write(
                    stage.join("quantize_index.json"),
                    b"{\"quarot_seed\":1,\"tensors\":[]}",
                )
                .unwrap();
                Ok(())
            },
            |_, _| {
                *called.borrow_mut() = true;
                Ok(passing_evidence())
            },
        )
        .unwrap();

        assert!(*called.borrow());
        assert!(!output.exists());
    }
}
