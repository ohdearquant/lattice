//! Produce comparator-v1 CPU greedy token dumps from a local Qwen3.5 checkpoint.
#![allow(clippy::field_reassign_with_default)]

use lattice_inference::tokenizer::common::{TokenizedInput, Tokenizer as _};
use lattice_inference::{GenerateConfig, GenerateOutput, StopReason};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    max_new_tokens: usize,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    prompt: String,
}
#[derive(Serialize)]
struct OutputCase<'a> {
    id: &'a str,
    prompt: &'a str,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    stop_reason: &'static str,
    complete: bool,
}
#[derive(Serialize)]
struct Dump<'a> {
    schema_version: u32,
    name: &'a str,
    config: serde_json::Value,
    cases: Vec<OutputCase<'a>>,
}

fn validate_manifest(m: &Manifest) -> Result<(), String> {
    if m.schema_version != 1 || m.max_new_tokens == 0 || m.cases.is_empty() {
        return Err("invalid manifest version, length, or empty cases".into());
    }
    let mut ids = HashSet::new();
    for c in &m.cases {
        if c.id.is_empty() || c.prompt.is_empty() || !ids.insert(&c.id) {
            return Err("empty or duplicate case id, or empty prompt".into());
        }
    }
    Ok(())
}
fn config(max: usize) -> Result<(GenerateConfig, serde_json::Value), String> {
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("../../quality_protocol_v1.json"))
            .map_err(|e| e.to_string())?;
    let expected = serde_json::json!({"mode":"greedy","stop_policy":"fixed-length-no-eos-or-stop","temperature":0.0,"top_k":1,"top_p":1.0,"min_p":0.0,"repetition_penalty":1.0,"stop_token_ids":[],"enable_thinking":false,"enable_mtp":false,"reasoning_budget":null,"stop_strings":[],"grammar":null});
    if v != expected {
        return Err("unsupported quality protocol".into());
    }
    v["max_new_tokens"] = serde_json::json!(max);
    let mut c = GenerateConfig::default();
    c.max_new_tokens = max;
    c.temperature = 0.0;
    c.top_k = 1;
    c.top_p = 1.0;
    c.min_p = 0.0;
    c.repetition_penalty = 1.0;
    c.seed = None;
    c.stop_token_ids.clear();
    c.enable_thinking = false;
    c.enable_mtp = Some(false);
    c.grammar = None;
    c.stop_strings.clear();
    c.reasoning_budget = None;
    c.logprobs = None;
    Ok((c, v))
}
fn produce<'a>(
    m: &'a Manifest,
    name: &'a str,
    mut tokenize: impl FnMut(&str) -> Result<TokenizedInput, String>,
    mut generate: impl FnMut(&str, &GenerateConfig) -> Result<GenerateOutput, String>,
) -> Result<Dump<'a>, String> {
    validate_manifest(m)?;
    if name.is_empty() {
        return Err("empty dump name".into());
    }
    let (cfg, config) = config(m.max_new_tokens)?;
    let mut cases = Vec::with_capacity(m.cases.len());
    for c in &m.cases {
        let t = tokenize(&c.prompt)?;
        if t.real_length == 0
            || t.pre_truncation_len != t.real_length
            || t.real_length > t.input_ids.len()
        {
            return Err(format!("case {}: empty or truncated prompt", c.id));
        }
        let prompt_ids = t.input_ids[..t.real_length].to_vec();
        let output = generate(&c.prompt, &cfg)?;
        if output.prompt_tokens != prompt_ids.len()
            || output.generated_tokens != m.max_new_tokens
            || output.token_ids.len() != m.max_new_tokens
            || output.stop_reason != Some(StopReason::Length)
            || output.stopped
        {
            return Err(format!("case {}: incomplete generation", c.id));
        }
        cases.push(OutputCase {
            id: &c.id,
            prompt: &c.prompt,
            prompt_ids,
            generated_ids: output.token_ids,
            stop_reason: "length",
            complete: true,
        });
    }
    Ok(Dump {
        schema_version: 1,
        name,
        config,
        cases,
    })
}
fn publish(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("invalid output path")?
        .to_string_lossy();
    for i in 0..100 {
        let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), i));
        let mut f = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        };
        let result = (|| -> std::io::Result<()> {
            f.write_all(bytes)?;
            f.sync_all()?;
            fs::hard_link(&temporary, path)?;
            Ok(())
        })();
        drop(f);
        let _ = fs::remove_file(&temporary);
        return result.map_err(|e| e.to_string());
    }
    Err("could not create temporary output".into())
}
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: quality_dump MODEL_DIR MANIFEST OUTPUT NAME".into());
    }
    let m: Manifest = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    validate_manifest(&m)?;
    if args[4].is_empty() {
        return Err("empty dump name".into());
    }
    let mut model =
        lattice_inference::model::qwen35::Qwen35Model::from_safetensors(Path::new(&args[1]))
            .map_err(|e| e.to_string())?;
    model.set_eos_token_id(u32::MAX);
    let dump = produce(
        &m,
        &args[4],
        |p| Ok(model.tokenizer().tokenize(p)),
        |p, c| model.generate(p, c).map_err(|e| e.to_string()),
    )?;
    let bytes = serde_json::to_vec_pretty(&dump).map_err(|e| e.to_string())?;
    publish(Path::new(&args[3]), &bytes)
}
fn main() {
    if let Err(e) = run() {
        eprintln!("quality dump refused: {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> Manifest {
        serde_json::from_str(
            r#"{"schema_version":1,"max_new_tokens":2,"cases":[{"id":"a","prompt":"hello"}]}"#,
        )
        .unwrap()
    }
    fn tokens() -> TokenizedInput {
        TokenizedInput {
            input_ids: vec![4, 5, 0],
            attention_mask: vec![1, 1, 0],
            token_type_ids: vec![0, 0, 0],
            real_length: 2,
            pre_truncation_len: 2,
        }
    }
    fn output() -> GenerateOutput {
        GenerateOutput {
            text: String::new(),
            token_ids: vec![6, 7],
            prompt_tokens: 2,
            generated_tokens: 2,
            stopped: false,
            stop_reason: Some(StopReason::Length),
            token_logprobs: vec![],
        }
    }
    #[test]
    fn records_real_prompt_slice_and_generated_tokens() {
        let m = manifest();
        let dump = produce(&m, "cpu", |_| Ok(tokens()), |_, _| Ok(output())).unwrap();
        assert_eq!(dump.cases[0].prompt_ids, vec![4, 5]);
        assert_eq!(dump.cases[0].generated_ids, vec![6, 7]);
        assert_eq!(dump.config["max_new_tokens"], 2);
    }
    #[test]
    fn rejects_truncation_before_generating() {
        let m = manifest();
        let result = produce(
            &m,
            "cpu",
            |_| {
                let mut t = tokens();
                t.pre_truncation_len = 3;
                Ok(t)
            },
            |_, _| panic!("must not generate"),
        );
        assert!(result.is_err());
    }
    #[test]
    fn rejects_early_stop_and_prompt_count_mismatch() {
        let m = manifest();
        assert!(
            produce(
                &m,
                "cpu",
                |_| Ok(tokens()),
                |_, _| {
                    let mut o = output();
                    o.stop_reason = Some(StopReason::Eos);
                    Ok(o)
                }
            )
            .is_err()
        );
        assert!(
            produce(
                &m,
                "cpu",
                |_| Ok(tokens()),
                |_, _| {
                    let mut o = output();
                    o.prompt_tokens = 1;
                    Ok(o)
                }
            )
            .is_err()
        );
        assert!(
            produce(
                &m,
                "cpu",
                |_| Ok(tokens()),
                |_, _| {
                    let mut o = output();
                    o.token_ids.pop();
                    Ok(o)
                }
            )
            .is_err()
        );
    }
    #[test]
    fn rejects_duplicate_ids_and_unknown_manifest_fields() {
        let mut m = manifest();
        m.cases.push(Case {
            id: "a".into(),
            prompt: "other".into(),
        });
        assert!(validate_manifest(&m).is_err());
        assert!(
            serde_json::from_str::<Manifest>(
                r#"{"schema_version":1,"max_new_tokens":2,"cases":[],"extra":0}"#
            )
            .is_err()
        );
    }
    #[test]
    fn publish_refuses_to_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dump.json");
        publish(&path, b"first").unwrap();
        assert!(publish(&path, b"second").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"first");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
