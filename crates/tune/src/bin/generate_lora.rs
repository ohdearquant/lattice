//! Qwen3.5 generation with LoRA adapter — proves PEFT/MLX-trained adapters work in Rust inference.
//!
//! Usage:
//!   cargo run --release -p lattice-tune --features "safetensors,inference-hook" \
//!     --bin generate_lora -- \
//!     --model-dir ~/.lattice/models/qwen3.5-0.8b \
//!     --lora adapter.safetensors \
//!     --prompt "Write a Rust function that checks if a number is prime" \
//!     --max-tokens 64
//!     [--json]   Emit @@lattice gen_token events for the Lattice Studio app.
//!
//! Optional flags that apply to a single prompt and to batch mode alike:
//!   --format raw|chat   `raw` (default) tokenizes the prompt as given. `chat` wraps it in
//!                       one user turn of the chat template and generates the assistant turn.
//!   --no-think          With `--format chat`: close an empty reasoning block in the prompt
//!                       and turn thinking off, so the reply starts with the answer.
//!   --grammar-file F    Constrain every generation with the GBNF grammar in file F.
//!
//! Batch mode (needs the `serde` feature, which the default feature set includes):
//!   --prompts-file P --out O [--resume]
//!     P is JSON Lines, one object per line with a string field `prompt` (other fields are
//!     ignored). The model and adapter load once. Each row is generated in file order and
//!     appended to O as one JSON line:
//!       {"idx":N,"output":"...","prompt_tokens":N,"generated_tokens":N,"stop_reason":"...","ms":N}
//!     A row whose generation fails is written as {"idx":N,"error":"...","ms":N} and the run
//!     continues; the exit status is 2 if any row failed. Without `--resume` an existing O is
//!     refused. With `--resume` every idx already in O is skipped and the rest are appended.
#![allow(clippy::field_reassign_with_default)]

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use lattice_inference::forward::metal_qwen35::{ChatMessage, format_chat_template};
use lattice_inference::grammar::{GrammarEngine, GrammarSpec};

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// The value after `flag`: `Ok(None)` when the flag is absent, an error when it is present
/// without a value (the end of the arguments, or another flag, follows it).
fn flag_value(args: &[String], flag: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == flag) {
        None => Ok(None),
        Some(i) => match args.get(i + 1) {
            Some(value) if !value.starts_with("--") => Ok(Some(value.clone())),
            _ => Err(format!("{flag} requires a value")),
        },
    }
}

fn parse_reasoning_budget(args: &[String]) -> Option<usize> {
    parse_arg(args, "--reasoning-budget")
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
}

/// An empty reasoning block, appended after the open assistant turn so the model answers
/// directly. `GenerateConfig::enable_thinking` does not prime the prompt itself.
const EMPTY_THINK_BLOCK: &str = "<think>\n\n</think>\n\n";

/// How a prompt string becomes the text the tokenizer sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptFormat {
    /// The prompt is tokenized exactly as given.
    Raw,
    /// The prompt is one user turn of the chat template, followed by the open assistant turn.
    Chat,
}

/// Settings shared by single-prompt and batch mode.
#[derive(Debug, PartialEq, Eq)]
struct GenOptions {
    format: PromptFormat,
    no_think: bool,
    grammar_file: Option<PathBuf>,
}

impl GenOptions {
    fn from_args(args: &[String]) -> Result<Self, String> {
        let format = match flag_value(args, "--format")?.as_deref() {
            None | Some("raw") => PromptFormat::Raw,
            Some("chat") => PromptFormat::Chat,
            Some(other) => return Err(format!("--format must be `raw` or `chat`, got `{other}`")),
        };
        let no_think = parse_flag(args, "--no-think");
        if no_think && format != PromptFormat::Chat {
            return Err("--no-think requires --format chat".to_string());
        }
        if no_think && parse_reasoning_budget(args).is_some() {
            return Err("--reasoning-budget has no effect with --no-think".to_string());
        }
        Ok(Self {
            format,
            no_think,
            grammar_file: flag_value(args, "--grammar-file")?.map(PathBuf::from),
        })
    }

    /// The text to generate from for `prompt`. `Raw` returns it unchanged.
    fn render(&self, prompt: &str) -> String {
        match self.format {
            PromptFormat::Raw => prompt.to_string(),
            PromptFormat::Chat => {
                let mut text = format_chat_template(&[ChatMessage::user(prompt)]);
                if self.no_think {
                    text.push_str(EMPTY_THINK_BLOCK);
                }
                text
            }
        }
    }
}

/// Compile GBNF text against the loaded model's tokenizer vocabulary.
fn build_grammar(
    model: &lattice_inference::model::qwen35::Qwen35Model,
    gbnf: String,
) -> Result<Arc<GrammarEngine>, String> {
    let vocab_bytes = model
        .tokenizer()
        .vocab_bytes(model.config().vocab_size)
        .map_err(|e| format!("tokenizer vocabulary unavailable for the grammar: {e}"))?;
    GrammarEngine::new(&GrammarSpec::Gbnf(gbnf), vocab_bytes)
        .map(Arc::new)
        .map_err(|e| format!("grammar failed to compile: {e}"))
}

/// Escape a string as a JSON string literal (including surrounding double quotes).
/// Does NOT depend on serde_json — this is a self-contained, correct escaper.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Other ASCII control characters: \u00XX
                let code = c as u32;
                out.push_str(&format!("\\u{code:04x}"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn default_model_cache() -> PathBuf {
    std::env::var("LATTICE_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".lattice").join("models")
        })
}

/// Batch mode: many prompts through one loaded model, one output JSON line per prompt.
/// Reading the input and checking a resumed output file need `serde_json`, an optional
/// dependency, so the whole module is behind the `serde` feature.
#[cfg(feature = "serde")]
mod batch {
    use std::collections::HashSet;
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    use lattice_inference::StopReason;
    use lattice_inference::model::qwen35::Qwen35Model;

    use super::{GenOptions, flag_value, json_escape, parse_flag};

    /// What the command line asks of batch mode.
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct BatchPlan {
        prompts_file: PathBuf,
        out: PathBuf,
        resume: bool,
    }

    impl BatchPlan {
        /// `Ok(None)` when no batch flag is present (single-prompt mode).
        pub(super) fn from_args(args: &[String]) -> Result<Option<Self>, String> {
            let prompts_file = flag_value(args, "--prompts-file")?;
            let out = flag_value(args, "--out")?;
            let resume = parse_flag(args, "--resume");
            let Some(prompts_file) = prompts_file else {
                if out.is_some() || resume {
                    return Err("--out and --resume require --prompts-file".to_string());
                }
                return Ok(None);
            };
            let out = out.ok_or("--prompts-file requires --out")?;
            if parse_flag(args, "--prompt") {
                return Err("--prompt cannot be combined with --prompts-file".to_string());
            }
            if parse_flag(args, "--json") {
                return Err(
                    "--json is for a single prompt and cannot be combined with --prompts-file"
                        .to_string(),
                );
            }
            Ok(Some(Self {
                prompts_file: PathBuf::from(prompts_file),
                out: PathBuf::from(out),
                resume,
            }))
        }
    }

    /// What an existing output file already holds.
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct ResumeScan {
        /// Row indices that already have an output line.
        done: HashSet<usize>,
        /// Byte length of the complete, newline-terminated lines. Anything after it is the
        /// remains of an interrupted write.
        valid_len: usize,
    }

    /// Everything read from disk before the model loads, so a bad input fails in seconds.
    pub(super) struct BatchJob {
        plan: BatchPlan,
        prompts: Vec<String>,
        resume: Option<ResumeScan>,
    }

    impl BatchJob {
        pub(super) fn load(plan: BatchPlan) -> Result<Self, String> {
            let text = std::fs::read_to_string(&plan.prompts_file)
                .map_err(|e| format!("cannot read {}: {e}", plan.prompts_file.display()))?;
            let prompts =
                read_prompts(&text).map_err(|e| format!("{}: {e}", plan.prompts_file.display()))?;
            let resume = if plan.out.exists() {
                if !plan.resume {
                    return Err(format!(
                        "{} already exists; pass --resume to continue it or choose another --out",
                        plan.out.display()
                    ));
                }
                let existing = std::fs::read_to_string(&plan.out)
                    .map_err(|e| format!("cannot read {}: {e}", plan.out.display()))?;
                Some(
                    scan_output(&existing, prompts.len())
                        .map_err(|e| format!("{}: {e}", plan.out.display()))?,
                )
            } else {
                None
            };
            Ok(Self {
                plan,
                prompts,
                resume,
            })
        }
    }

    /// The `prompt` of every line of a JSON Lines file, in file order. Row `idx` is the line
    /// number minus one; fields other than `prompt` are ignored. A blank line is an error.
    pub(super) fn read_prompts(text: &str) -> Result<Vec<String>, String> {
        let mut prompts = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line_no = i + 1;
            let row: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("line {line_no}: not valid JSON: {e}"))?;
            let Some(object) = row.as_object() else {
                return Err(format!("line {line_no}: row is not a JSON object"));
            };
            match object.get("prompt") {
                Some(serde_json::Value::String(prompt)) => prompts.push(prompt.clone()),
                Some(_) => return Err(format!("line {line_no}: `prompt` is not a string")),
                None => return Err(format!("line {line_no}: row has no `prompt` field")),
            }
        }
        if prompts.is_empty() {
            return Err("no prompt rows".to_string());
        }
        Ok(prompts)
    }

    /// Row indices already present in an earlier output file. A final line without its
    /// newline is the remains of an interrupted write: it is not counted and `valid_len`
    /// stops before it. Any other unreadable line is an error, as is an `idx` that the
    /// prompts file (`row_count` rows) cannot have produced.
    pub(super) fn scan_output(text: &str, row_count: usize) -> Result<ResumeScan, String> {
        let mut done = HashSet::new();
        let mut valid_len = 0;
        for (i, segment) in text.split_inclusive('\n').enumerate() {
            if !segment.ends_with('\n') {
                break;
            }
            let line_no = i + 1;
            let row: serde_json::Value = serde_json::from_str(segment)
                .map_err(|e| format!("line {line_no}: not valid JSON: {e}"))?;
            let idx = row
                .get("idx")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("line {line_no}: row has no integer `idx`"))?;
            if idx >= row_count {
                return Err(format!(
                    "line {line_no}: idx {idx} is outside the {row_count} rows of the prompts file"
                ));
            }
            done.insert(idx);
            valid_len += segment.len();
        }
        Ok(ResumeScan { done, valid_len })
    }

    pub(super) fn encode_ok_row(
        idx: usize,
        output: &str,
        prompt_tokens: usize,
        generated_tokens: usize,
        stop_reason: &str,
        ms: u128,
    ) -> String {
        format!(
            "{{\"idx\":{idx},\"output\":{},\"prompt_tokens\":{prompt_tokens},\"generated_tokens\":{generated_tokens},\"stop_reason\":{},\"ms\":{ms}}}",
            json_escape(output),
            json_escape(stop_reason)
        )
    }

    pub(super) fn encode_error_row(idx: usize, error: &str, ms: u128) -> String {
        format!(
            "{{\"idx\":{idx},\"error\":{},\"ms\":{ms}}}",
            json_escape(error)
        )
    }

    pub(super) fn stop_reason_name(reason: Option<StopReason>) -> &'static str {
        match reason {
            Some(StopReason::Eos) => "eos",
            Some(StopReason::Grammar) => "grammar",
            Some(StopReason::Length) => "length",
            Some(StopReason::KvFull) => "kv_full",
            Some(StopReason::Interrupt) => "interrupt",
            Some(_) => "other",
            None => "none",
        }
    }

    /// Open the output file for appending. A resumed file is first cut back to its last
    /// complete line; a new file must not exist yet.
    fn open_output(plan: &BatchPlan, resume: Option<&ResumeScan>) -> Result<File, String> {
        let describe = |e: std::io::Error| format!("cannot open {}: {e}", plan.out.display());
        match resume {
            Some(scan) => {
                let file = OpenOptions::new()
                    .append(true)
                    .open(&plan.out)
                    .map_err(describe)?;
                let valid_len = scan.valid_len as u64;
                if file.metadata().map_err(describe)?.len() != valid_len {
                    file.set_len(valid_len).map_err(describe)?;
                }
                Ok(file)
            }
            None => OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&plan.out)
                .map_err(|e| {
                    if e.kind() == std::io::ErrorKind::AlreadyExists {
                        format!(
                            "{} already exists; pass --resume to continue it or choose another --out",
                            plan.out.display()
                        )
                    } else {
                        describe(e)
                    }
                }),
        }
    }

    /// Generate every row that is not already in the output, appending one flushed line per
    /// row. Returns how many rows failed to generate.
    pub(super) fn run(
        job: BatchJob,
        model: &Qwen35Model,
        gen_cfg: &lattice_inference::GenerateConfig,
        opts: &GenOptions,
    ) -> Result<usize, String> {
        let mut out = open_output(&job.plan, job.resume.as_ref())?;
        let total = job.prompts.len();
        if let Some(scan) = &job.resume {
            println!(
                "Resuming: {} of {total} rows already in {}",
                scan.done.len(),
                job.plan.out.display()
            );
        }
        let mut errored = 0;
        for (idx, prompt) in job.prompts.iter().enumerate() {
            if job
                .resume
                .as_ref()
                .is_some_and(|scan| scan.done.contains(&idx))
            {
                continue;
            }
            let started = Instant::now();
            let result = model.generate(&opts.render(prompt), gen_cfg);
            let ms = started.elapsed().as_millis();
            let mut line = match result {
                Ok(output) => {
                    println!(
                        "[{}/{total}] {} tokens in {ms}ms",
                        idx + 1,
                        output.generated_tokens
                    );
                    encode_ok_row(
                        idx,
                        &output.text,
                        output.prompt_tokens,
                        output.generated_tokens,
                        stop_reason_name(output.stop_reason),
                        ms,
                    )
                }
                Err(e) => {
                    errored += 1;
                    eprintln!("[{}/{total}] generation failed: {e}", idx + 1);
                    encode_error_row(idx, &e.to_string(), ms)
                }
            };
            line.push('\n');
            out.write_all(line.as_bytes())
                .and_then(|()| out.flush())
                .map_err(|e| format!("cannot write {}: {e}", job.plan.out.display()))?;
        }
        Ok(errored)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn args(list: &[&str]) -> Vec<String> {
            list.iter().map(ToString::to_string).collect()
        }

        fn plan(out: &std::path::Path, resume: bool) -> BatchPlan {
            BatchPlan {
                prompts_file: PathBuf::from("unused.jsonl"),
                out: out.to_path_buf(),
                resume,
            }
        }

        #[test]
        fn batch_plan_is_absent_without_batch_flags() {
            assert_eq!(
                BatchPlan::from_args(&args(&["bin", "--prompt", "hi"])),
                Ok(None)
            );
        }

        #[test]
        fn batch_plan_reads_paths_and_resume() {
            let parsed = BatchPlan::from_args(&args(&[
                "bin",
                "--prompts-file",
                "in.jsonl",
                "--out",
                "out.jsonl",
                "--resume",
            ]))
            .unwrap()
            .unwrap();
            assert_eq!(parsed.prompts_file, PathBuf::from("in.jsonl"));
            assert_eq!(parsed.out, PathBuf::from("out.jsonl"));
            assert!(parsed.resume);
            let without = BatchPlan::from_args(&args(&[
                "bin",
                "--prompts-file",
                "in.jsonl",
                "--out",
                "out.jsonl",
            ]))
            .unwrap()
            .unwrap();
            assert!(!without.resume);
        }

        #[test]
        fn batch_plan_rejects_inconsistent_flags() {
            let missing_out = BatchPlan::from_args(&args(&["bin", "--prompts-file", "in.jsonl"]));
            assert!(missing_out.unwrap_err().contains("requires --out"));
            for stray in [&["bin", "--out", "o.jsonl"][..], &["bin", "--resume"][..]] {
                let err = BatchPlan::from_args(&args(stray)).unwrap_err();
                assert!(err.contains("require --prompts-file"), "{err}");
            }
            for single in ["--prompt", "--json"] {
                let err = BatchPlan::from_args(&args(&[
                    "bin",
                    "--prompts-file",
                    "in.jsonl",
                    "--out",
                    "o.jsonl",
                    single,
                    "x",
                ]))
                .unwrap_err();
                assert!(err.contains(single), "{err}");
            }
            let valueless = BatchPlan::from_args(&args(&["bin", "--prompts-file"]));
            assert!(
                valueless
                    .unwrap_err()
                    .contains("--prompts-file requires a value")
            );
        }

        #[test]
        fn read_prompts_keeps_order_and_ignores_other_fields() {
            let text = concat!(
                "{\"prompt\":\"first\",\"label\":{\"nested\":[1,2]}}\r\n",
                "{\"extra\":1,\"prompt\":\"say \\\"hi\\\"\\nthen \\\\ done\"}\n",
            );
            assert_eq!(
                read_prompts(text).unwrap(),
                vec!["first".to_string(), "say \"hi\"\nthen \\ done".to_string()]
            );
        }

        #[test]
        fn read_prompts_names_the_one_based_line_of_a_bad_row() {
            let text = "{\"prompt\":\"a\"}\n{\"prompt\":\"b\"}\n{\"prompt\": \n";
            let err = read_prompts(text).unwrap_err();
            assert!(err.starts_with("line 3:"), "{err}");
        }

        #[test]
        fn read_prompts_rejects_rows_without_a_string_prompt() {
            let cases = [
                (
                    "{\"prompt\":\"a\"}\n{\"other\":1}\n",
                    "line 2: row has no `prompt`",
                ),
                ("{\"prompt\":7}\n", "line 1: `prompt` is not a string"),
                (
                    "{\"prompt\":\"a\"}\n[\"prompt\"]\n",
                    "line 2: row is not a JSON object",
                ),
                (
                    "{\"prompt\":\"a\"}\n\n{\"prompt\":\"b\"}\n",
                    "line 2: not valid JSON",
                ),
            ];
            for (text, expected) in cases {
                let err = read_prompts(text).unwrap_err();
                assert!(err.starts_with(expected), "{text:?} gave {err}");
            }
            assert!(read_prompts("").is_err());
        }

        #[test]
        fn scan_output_collects_done_rows_and_drops_a_torn_final_line() {
            let first = "{\"idx\":0,\"output\":\"a\"}\n";
            let third = "{\"idx\":2,\"error\":\"boom\",\"ms\":3}\n";
            let torn = "{\"idx\":1,\"output\":\"par";
            let scan = scan_output(&format!("{first}{third}{torn}"), 3).unwrap();
            assert_eq!(scan.done, HashSet::from([0, 2]));
            assert_eq!(scan.valid_len, first.len() + third.len());
        }

        #[test]
        fn scan_output_treats_a_complete_row_missing_its_newline_as_torn() {
            let first = "{\"idx\":0}\n";
            let scan = scan_output(&format!("{first}{{\"idx\":1}}"), 2).unwrap();
            assert_eq!(scan.done, HashSet::from([0]));
            assert_eq!(scan.valid_len, first.len());
        }

        #[test]
        fn scan_output_rejects_unreadable_lines_with_their_number() {
            let err = scan_output("{\"idx\":0}\nnot json\n", 2).unwrap_err();
            assert!(err.starts_with("line 2:"), "{err}");
            let err = scan_output("{\"idx\":0}\n{\"output\":\"x\"}\n", 2).unwrap_err();
            assert!(err.starts_with("line 2: row has no integer `idx`"), "{err}");
        }

        #[test]
        fn scan_output_rejects_an_idx_the_prompts_file_cannot_have_produced() {
            let err = scan_output("{\"idx\":0}\n{\"idx\":2}\n", 2).unwrap_err();
            assert!(
                err.starts_with("line 2: idx 2 is outside the 2 rows"),
                "{err}"
            );
            assert!(scan_output("{\"idx\":1}\n", 2).is_ok());
        }

        #[test]
        fn encoded_ok_row_round_trips_text_that_needs_escaping() {
            let output = "quote \" backslash \\ C:\\temp\\new\nline\ttab\u{1}ctl é✓";
            let line = encode_ok_row(4, output, 11, 7, "length", 1234);
            assert!(!line.contains('\n'), "a row must stay on one line");
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(value["idx"], 4);
            assert_eq!(value["output"], output);
            assert_eq!(value["prompt_tokens"], 11);
            assert_eq!(value["generated_tokens"], 7);
            assert_eq!(value["stop_reason"], "length");
            assert_eq!(value["ms"], 1234);
            assert_eq!(value.as_object().unwrap().len(), 6);
        }

        #[test]
        fn encoded_error_row_carries_the_error_and_no_output() {
            let line = encode_error_row(9, "bad \"thing\"\nhappened", 5);
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(value["idx"], 9);
            assert_eq!(value["error"], "bad \"thing\"\nhappened");
            assert_eq!(value["ms"], 5);
            assert!(value.get("output").is_none());
        }

        #[test]
        fn stop_reasons_have_distinct_stable_names() {
            let names = [
                stop_reason_name(Some(StopReason::Eos)),
                stop_reason_name(Some(StopReason::Grammar)),
                stop_reason_name(Some(StopReason::Length)),
                stop_reason_name(Some(StopReason::KvFull)),
                stop_reason_name(Some(StopReason::Interrupt)),
                stop_reason_name(None),
            ];
            assert_eq!(
                names,
                ["eos", "grammar", "length", "kv_full", "interrupt", "none"]
            );
        }

        #[test]
        fn load_refuses_an_existing_output_unless_resuming() {
            let dir = tempfile::tempdir().unwrap();
            let prompts = dir.path().join("in.jsonl");
            let out = dir.path().join("out.jsonl");
            std::fs::write(&prompts, "{\"prompt\":\"a\"}\n{\"prompt\":\"b\"}\n").unwrap();
            std::fs::write(&out, "{\"idx\":1,\"output\":\"x\"}\n").unwrap();
            let plan_for = |resume| BatchPlan {
                prompts_file: prompts.clone(),
                out: out.clone(),
                resume,
            };
            let err = BatchJob::load(plan_for(false)).err().unwrap();
            assert!(err.contains("already exists"), "{err}");
            let job = BatchJob::load(plan_for(true)).unwrap();
            assert_eq!(job.prompts, vec!["a".to_string(), "b".to_string()]);
            assert_eq!(job.resume.unwrap().done, HashSet::from([1]));
            assert_eq!(
                std::fs::read_to_string(&out).unwrap(),
                "{\"idx\":1,\"output\":\"x\"}\n",
                "loading must not modify the existing output"
            );
        }

        #[test]
        fn load_without_an_output_file_starts_fresh() {
            let dir = tempfile::tempdir().unwrap();
            let prompts = dir.path().join("in.jsonl");
            std::fs::write(&prompts, "{\"prompt\":\"a\"}\n").unwrap();
            for resume in [false, true] {
                let job = BatchJob::load(BatchPlan {
                    prompts_file: prompts.clone(),
                    out: dir.path().join("fresh.jsonl"),
                    resume,
                })
                .unwrap();
                assert!(job.resume.is_none());
            }
        }

        #[test]
        fn open_output_cuts_a_torn_tail_before_appending() {
            let dir = tempfile::tempdir().unwrap();
            let out = dir.path().join("out.jsonl");
            let complete = "{\"idx\":0,\"output\":\"a\"}\n";
            std::fs::write(&out, format!("{complete}{{\"idx\":1,\"outp")).unwrap();
            let scan = scan_output(&std::fs::read_to_string(&out).unwrap(), 2).unwrap();
            let mut file = open_output(&plan(&out, true), Some(&scan)).unwrap();
            file.write_all(b"{\"idx\":1,\"output\":\"b\"}\n").unwrap();
            assert_eq!(
                std::fs::read_to_string(&out).unwrap(),
                format!("{complete}{{\"idx\":1,\"output\":\"b\"}}\n")
            );
        }

        #[test]
        fn open_output_creates_new_but_never_overwrites() {
            let dir = tempfile::tempdir().unwrap();
            let out = dir.path().join("out.jsonl");
            open_output(&plan(&out, false), None).unwrap();
            assert!(out.exists());
            std::fs::write(&out, "keep me\n").unwrap();
            let err = open_output(&plan(&out, false), None).unwrap_err();
            assert!(err.contains("already exists"), "{err}");
            assert_eq!(std::fs::read_to_string(&out).unwrap(), "keep me\n");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let opts = match GenOptions::from_args(&args) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    #[cfg(feature = "serde")]
    let batch_job = match batch::BatchPlan::from_args(&args)
        .and_then(|plan| plan.map(batch::BatchJob::load).transpose())
    {
        Ok(job) => job,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    #[cfg(not(feature = "serde"))]
    {
        if ["--prompts-file", "--out", "--resume"]
            .iter()
            .any(|flag| parse_flag(&args, flag))
        {
            eprintln!(
                "batch mode needs the `serde` feature: rebuild with --features safetensors,inference-hook,serde"
            );
            std::process::exit(1);
        }
    }
    let grammar_text = opts.grammar_file.as_ref().map(|path| {
        std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read grammar file {path:?}: {e}");
            std::process::exit(1);
        })
    });

    let prompt =
        parse_arg(&args, "--prompt").unwrap_or_else(|| "What is the meaning of life?".to_string());

    let max_tokens: usize = parse_arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let seed: Option<u64> = parse_arg(&args, "--seed").and_then(|s| s.parse().ok());

    let temperature: Option<f32> = parse_arg(&args, "--temperature").and_then(|s| s.parse().ok());

    let top_k: Option<usize> = parse_arg(&args, "--top-k").and_then(|s| s.parse().ok());
    let top_p: Option<f32> = parse_arg(&args, "--top-p").and_then(|s| s.parse().ok());
    let repetition_penalty: Option<f32> =
        parse_arg(&args, "--repetition-penalty").and_then(|s| s.parse().ok());
    let reasoning_budget = parse_reasoning_budget(&args);

    let lora_path: Option<PathBuf> = parse_arg(&args, "--lora").map(PathBuf::from);

    let emit_json = parse_flag(&args, "--json");

    let model_dir = if let Some(dir) = parse_arg(&args, "--model-dir") {
        PathBuf::from(dir)
    } else {
        let model_name = parse_arg(&args, "--model").unwrap_or_else(|| "qwen3.5-0.8b".to_string());
        default_model_cache().join(model_name)
    };

    // Load base model
    println!("Loading model from {model_dir:?}...");
    let t0 = Instant::now();

    let mut model =
        match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(&model_dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        };

    println!("Model loaded in {}ms", t0.elapsed().as_millis());

    // Load LoRA adapter
    if let Some(ref lora) = lora_path {
        println!("Loading LoRA adapter from {lora:?}...");
        let t_lora = Instant::now();

        match lattice_tune::lora::LoraAdapter::from_safetensors(lora) {
            Ok(adapter) => {
                if let Err(e) = adapter.validate_against(model.config()) {
                    eprintln!("LoRA adapter incompatible with loaded model: {e}");
                    std::process::exit(1);
                }
                println!(
                    "  Adapter: {} pairs, rank={}, scale={:.2}, {} parameters",
                    adapter.num_adapted_layers(),
                    adapter.config().rank,
                    adapter.config().scale(),
                    adapter.num_parameters()
                );
                if let Err(e) = model.set_lora(Box::new(adapter)) {
                    eprintln!("LoRA adapter incompatible with loaded model: {e}");
                    std::process::exit(1);
                }
                println!(
                    "  LoRA active (loaded in {}ms)",
                    t_lora.elapsed().as_millis()
                );
            }
            Err(e) => {
                eprintln!("Failed to load LoRA adapter: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("No LoRA adapter (base model only)");
    }

    // Configure generation
    let mut gen_cfg = lattice_inference::GenerateConfig::default();
    gen_cfg.max_new_tokens = max_tokens;
    gen_cfg.seed = seed;
    if let Some(t) = temperature {
        gen_cfg.temperature = t;
    }
    if let Some(k) = top_k {
        gen_cfg.top_k = k;
    }
    if let Some(p) = top_p {
        gen_cfg.top_p = p;
    }
    if let Some(r) = repetition_penalty {
        gen_cfg.repetition_penalty = r;
    }
    if reasoning_budget.is_some() {
        gen_cfg.reasoning_budget = reasoning_budget;
    }
    if opts.no_think {
        gen_cfg.enable_thinking = false;
    }
    if let Some(gbnf) = grammar_text {
        match build_grammar(&model, gbnf) {
            Ok(engine) => gen_cfg.grammar = Some(engine),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }

    // Batch and chat prompts are admitted up to the model's context window; the tokenizer's
    // default cap would otherwise cut a long prompt short without any error.
    #[cfg(feature = "serde")]
    {
        if let Some(job) = batch_job {
            model.ensure_tokenizer_max_seq_len(model.max_context());
            match batch::run(job, &model, &gen_cfg, &opts) {
                Ok(0) => return,
                Ok(errored) => {
                    eprintln!(
                        "{errored} row(s) failed to generate; see the `error` fields in the output"
                    );
                    std::process::exit(2);
                }
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
    }
    if opts.format == PromptFormat::Chat {
        model.ensure_tokenizer_max_seq_len(model.max_context());
    }
    let gen_prompt = opts.render(&prompt);

    println!("\nPrompt: {prompt}");
    if opts.format == PromptFormat::Chat {
        println!(
            "Format: chat{}",
            if opts.no_think { " (no-think)" } else { "" }
        );
    }
    if let Some(path) = &opts.grammar_file {
        println!("Grammar: {path:?}");
    }
    println!(
        "Config: temp={}, top_k={}, top_p={}, rep_penalty={}, seed={:?}, max_tokens={}",
        gen_cfg.temperature,
        gen_cfg.top_k,
        gen_cfg.top_p,
        gen_cfg.repetition_penalty,
        gen_cfg.seed,
        max_tokens
    );
    println!("Generating...\n");

    let t1 = Instant::now();

    if emit_json {
        // Streaming JSON mode: emit @@lattice gen_token events live.
        let mut stdout = std::io::stdout();
        let mut first_token_emitted = false;
        let mut ttft_ms: f64 = 0.0;

        let result = model.generate_streaming(&gen_prompt, &gen_cfg, |delta| {
            if !first_token_emitted {
                ttft_ms = t1.elapsed().as_secs_f64() * 1000.0;
                first_token_emitted = true;
            }
            let token_json = json_escape(delta);
            writeln!(
                stdout,
                "@@lattice {{\"ev\":\"gen_token\",\"token\":{token_json},\"done\":false}}"
            )
            .ok();
            stdout.flush().ok();
        });

        match result {
            Ok(output) => {
                let gen_ms = t1.elapsed().as_millis();
                let tok_s = if gen_ms > 0 {
                    output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
                } else {
                    0.0
                };
                // Final done event with stats. prompt_tokens / gen_tokens / total_ms are
                // additive and kept byte-compatible with chat_metal's done event.
                writeln!(
                    stdout,
                    "@@lattice {{\"ev\":\"gen_token\",\"token\":\"\",\"done\":true,\"tok_s\":{tok_s:.1},\"ttft_ms\":{ttft_ms:.1},\"prompt_tokens\":{},\"gen_tokens\":{},\"total_ms\":{}}}",
                    output.prompt_tokens, output.generated_tokens, gen_ms
                )
                .ok();
                stdout.flush().ok();
            }
            Err(e) => {
                eprintln!("Generation failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // Non-JSON mode: original atomic output block, unchanged.
        match model.generate(&gen_prompt, &gen_cfg) {
            Ok(output) => {
                let gen_ms = t1.elapsed().as_millis();
                let tok_s = if gen_ms > 0 {
                    output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
                } else {
                    0.0
                };

                println!("--- Output ---");
                println!("{}", output.text);
                println!("--- Stats ---");
                println!(
                    "Tokens: {} prompt + {} generated in {}ms ({:.1} tok/s)",
                    output.prompt_tokens, output.generated_tokens, gen_ms, tok_s
                );
                if lora_path.is_some() {
                    println!("LoRA: ACTIVE");
                }
            }
            Err(e) => {
                eprintln!("Generation failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_inference::{BpeTokenizer, Tokenizer};

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(ToString::to_string).collect()
    }

    fn chat(no_think: bool) -> GenOptions {
        GenOptions {
            format: PromptFormat::Chat,
            no_think,
            grammar_file: None,
        }
    }

    #[test]
    fn options_default_to_raw_without_grammar() {
        let opts = GenOptions::from_args(&args(&["bin", "--prompt", "hi"])).unwrap();
        assert_eq!(
            opts,
            GenOptions {
                format: PromptFormat::Raw,
                no_think: false,
                grammar_file: None,
            }
        );
    }

    #[test]
    fn options_parse_chat_no_think_and_grammar_file() {
        let opts = GenOptions::from_args(&args(&[
            "bin",
            "--format",
            "chat",
            "--no-think",
            "--grammar-file",
            "g.gbnf",
        ]))
        .unwrap();
        assert_eq!(opts.format, PromptFormat::Chat);
        assert!(opts.no_think);
        assert_eq!(opts.grammar_file, Some(PathBuf::from("g.gbnf")));
        let raw = GenOptions::from_args(&args(&["bin", "--format", "raw"])).unwrap();
        assert_eq!(raw.format, PromptFormat::Raw);
    }

    #[test]
    fn options_reject_an_unknown_format() {
        let err = GenOptions::from_args(&args(&["bin", "--format", "markdown"])).unwrap_err();
        assert!(
            err.contains("`raw` or `chat`") && err.contains("markdown"),
            "{err}"
        );
    }

    #[test]
    fn options_reject_no_think_outside_chat_format() {
        for list in [
            &["bin", "--no-think"][..],
            &["bin", "--format", "raw", "--no-think"][..],
        ] {
            let err = GenOptions::from_args(&args(list)).unwrap_err();
            assert!(err.contains("--no-think requires --format chat"), "{err}");
        }
    }

    #[test]
    fn options_reject_a_reasoning_budget_that_no_think_would_ignore() {
        let conflicting = args(&[
            "bin",
            "--format",
            "chat",
            "--no-think",
            "--reasoning-budget",
            "64",
        ]);
        let err = GenOptions::from_args(&conflicting).unwrap_err();
        assert!(err.contains("--reasoning-budget"), "{err}");
        let zero_budget = args(&[
            "bin",
            "--format",
            "chat",
            "--no-think",
            "--reasoning-budget",
            "0",
        ]);
        assert!(GenOptions::from_args(&zero_budget).is_ok());
    }

    #[test]
    fn options_reject_a_flag_missing_its_value() {
        for list in [
            &["bin", "--format"][..],
            &["bin", "--format", "--no-think"][..],
            &["bin", "--grammar-file"][..],
        ] {
            let err = GenOptions::from_args(&args(list)).unwrap_err();
            assert!(err.contains("requires a value"), "{err}");
        }
    }

    #[test]
    fn raw_render_returns_the_prompt_unchanged() {
        let opts = GenOptions::from_args(&args(&["bin"])).unwrap();
        let prompt = "  keep <|im_start|>as is\n<think> ";
        assert_eq!(opts.render(prompt), prompt);
    }

    #[test]
    fn chat_render_is_one_user_turn_and_an_open_assistant_turn() {
        assert_eq!(
            chat(false).render("What is 2+2?"),
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chat_render_with_no_think_closes_an_empty_reasoning_block() {
        assert_eq!(
            chat(true).render("What is 2+2?"),
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// A tokenizer.json with the added-token ids and `special` flags of the Qwen3.5
    /// checkpoints, over a vocabulary just large enough to spell the test strings.
    /// `Ċ` is the byte-level spelling of a newline.
    const QWEN_STYLE_TOKENIZER: &str = r#"{
        "model": {
            "type": "BPE",
            "vocab": {"u":0,"s":1,"e":2,"r":3,"h":4,"i":5,"a":6,"t":7,"n":8,"Ċ":9},
            "merges": []
        },
        "added_tokens": [
            {"id": 248045, "content": "<|im_start|>", "special": true},
            {"id": 248046, "content": "<|im_end|>", "special": true},
            {"id": 248068, "content": "<think>", "special": false},
            {"id": 248069, "content": "</think>", "special": false}
        ]
    }"#;

    fn encode(tokenizer: &BpeTokenizer, text: &str) -> Vec<u32> {
        let input = tokenizer.tokenize(text);
        input.input_ids[..input.real_length].to_vec()
    }

    #[test]
    fn raw_prompt_markers_encode_as_single_special_ids() {
        let tokenizer = BpeTokenizer::from_tokenizer_json_str(QWEN_STYLE_TOKENIZER).unwrap();
        assert_eq!(
            encode(&tokenizer, "<|im_start|>hi<|im_end|><think></think>"),
            vec![248045, 4, 5, 248046, 248068, 248069]
        );
    }

    #[test]
    fn rendered_chat_prompt_encodes_its_markers_as_special_ids() {
        let tokenizer = BpeTokenizer::from_tokenizer_json_str(QWEN_STYLE_TOKENIZER).unwrap();
        let ids = encode(&tokenizer, &chat(true).render("hi"));
        let (start, end, open, close) = (248045, 248046, 248068, 248069);
        let (u, s, e, r, h, i, a, t, n, nl) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(
            ids,
            vec![
                start, u, s, e, r, nl, h, i, end, nl, start, a, s, s, i, s, t, a, n, t, nl, open,
                nl, nl, close, nl, nl
            ]
        );
    }
}
