use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

use khive_request::{parse_request, ArgValue, ExecutionMode, ParsedOp, ParsedRequest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const PARSER: &str = "khive_request::parse_request";
const SOURCE_SHA: &str = env!("KHIVE_PARSER_SOURCE_SHA");

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum WireArg {
    Value { value: Value },
    PrevRef { path: String },
    Array { items: Vec<WireArg> },
    Object { entries: Vec<(String, WireArg)> },
}

impl From<&ArgValue> for WireArg {
    fn from(arg: &ArgValue) -> Self {
        match arg {
            ArgValue::Value(value) => Self::Value {
                value: value.clone(),
            },
            ArgValue::PrevRef { path } => Self::PrevRef { path: path.clone() },
            ArgValue::Array(items) => Self::Array {
                items: items.iter().map(Self::from).collect(),
            },
            ArgValue::Object(entries) => Self::Object {
                entries: entries
                    .iter()
                    .map(|(key, value)| (key.clone(), Self::from(value)))
                    .collect(),
            },
        }
    }
}

impl From<WireArg> for ArgValue {
    fn from(arg: WireArg) -> Self {
        match arg {
            WireArg::Value { value } => Self::Value(value),
            WireArg::PrevRef { path } => Self::PrevRef { path },
            WireArg::Array { items } => Self::Array(items.into_iter().map(Self::from).collect()),
            WireArg::Object { entries } => Self::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum WireMode {
    Single,
    Parallel,
    Chain,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WireOp {
    tool: String,
    args: BTreeMap<String, WireArg>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WireRequest {
    mode: WireMode,
    ops: Vec<WireOp>,
}

impl From<&ParsedRequest> for WireRequest {
    fn from(request: &ParsedRequest) -> Self {
        Self {
            mode: match request.mode {
                ExecutionMode::Single => WireMode::Single,
                ExecutionMode::Parallel => WireMode::Parallel,
                ExecutionMode::Chain => WireMode::Chain,
            },
            ops: request
                .ops
                .iter()
                .map(|op| WireOp {
                    tool: op.tool.clone(),
                    args: op
                        .args
                        .iter()
                        .map(|(key, value)| (key.clone(), WireArg::from(value)))
                        .collect(),
                })
                .collect(),
        }
    }
}

impl From<WireRequest> for ParsedRequest {
    fn from(request: WireRequest) -> Self {
        Self {
            mode: match request.mode {
                WireMode::Single => ExecutionMode::Single,
                WireMode::Parallel => ExecutionMode::Parallel,
                WireMode::Chain => ExecutionMode::Chain,
            },
            ops: request
                .ops
                .into_iter()
                .map(|op| ParsedOp {
                    tool: op.tool,
                    args: op
                        .args
                        .into_iter()
                        .map(|(key, value)| (key, ArgValue::from(value)))
                        .collect(),
                })
                .collect(),
        }
    }
}

fn render_arg(arg: &ArgValue) -> Result<String, String> {
    Ok(match arg {
        ArgValue::Value(value) => render_value(value)?,
        ArgValue::PrevRef { path } => {
            // Quoted references preserve upstream path spelling, including the
            // distinction between parsed bare and quoted array-index paths.
            let expression = if path.is_empty() {
                "$prev".to_owned()
            } else {
                format!("$prev.{path}")
            };
            serde_json::to_string(&expression).map_err(|e| e.to_string())?
        }
        ArgValue::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(render_arg)
                .collect::<Result<Vec<_>, _>>()?
                .join(",")
        ),
        ArgValue::Object(entries) => {
            let fields = entries
                .iter()
                .map(|(key, value)| {
                    Ok(format!(
                        "{}:{}",
                        serde_json::to_string(key).map_err(|e| e.to_string())?,
                        render_arg(value)?
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            format!("{{{}}}", fields.join(","))
        }
    })
}

fn render_value(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text)
            if text == "$prev" || text.starts_with("$prev.") || text.starts_with("$prev[") =>
        {
            serde_json::to_string(&format!("\\{text}")).map_err(|e| e.to_string())
        }
        Value::Array(items) => Ok(format!(
            "[{}]",
            items
                .iter()
                .map(render_value)
                .collect::<Result<Vec<_>, _>>()?
                .join(",")
        )),
        Value::Object(fields) => {
            let pairs = fields
                .iter()
                .map(|(key, value)| {
                    Ok(format!(
                        "{}:{}",
                        serde_json::to_string(key).map_err(|e| e.to_string())?,
                        render_value(value)?
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(format!("{{{}}}", pairs.join(",")))
        }
        _ => serde_json::to_string(value).map_err(|e| e.to_string()),
    }
}

fn render_functions(request: &ParsedRequest) -> Result<String, String> {
    let ops = request
        .ops
        .iter()
        .map(|op| {
            let args = op
                .args
                .iter()
                .map(|(key, value)| Ok(format!("{key}={}", render_arg(value)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(format!("{}({})", op.tool, args.join(",")))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(match request.mode {
        ExecutionMode::Single => ops.join(""),
        ExecutionMode::Parallel => format!("[{}]", ops.join(",")),
        ExecutionMode::Chain => ops.join(" | "),
    })
}

fn render_json_request(request: &ParsedRequest) -> Result<String, String> {
    let ops = request
        .ops
        .iter()
        .map(|op| {
            let args = op
                .args
                .iter()
                .map(|(key, value)| {
                    value
                        .as_value()
                        .cloned()
                        .map(|value| (key.clone(), value))
                        .ok_or_else(|| "JSON request form cannot represent references".to_owned())
                })
                .collect::<Result<serde_json::Map<_, _>, _>>()?;
            Ok(json!({"tool": op.tool, "args": args}))
        })
        .collect::<Result<Vec<_>, String>>()?;
    match request.mode {
        ExecutionMode::Single => serde_json::to_string(&ops[0]).map_err(|e| e.to_string()),
        ExecutionMode::Parallel => serde_json::to_string(&ops).map_err(|e| e.to_string()),
        ExecutionMode::Chain => Err("JSON request form cannot represent chain mode".to_owned()),
    }
}

fn validate(completion: &str) -> Result<Value, String> {
    let parsed = parse_request(completion).map_err(|e| format!("parse: {e}"))?;
    let wire = WireRequest::from(&parsed);
    let encoded = serde_json::to_vec(&wire).map_err(|e| e.to_string())?;
    let restored: WireRequest = serde_json::from_slice(&encoded).map_err(|e| e.to_string())?;
    if ParsedRequest::from(restored) != parsed {
        return Err("AST JSON roundtrip changed the parsed structure".to_owned());
    }
    let (canonical_format, canonical_request) = if matches!(parsed.mode, ExecutionMode::Chain) {
        ("function_call_dsl", render_functions(&parsed)?)
    } else {
        let json_request = render_json_request(&parsed)?;
        match parse_request(&json_request) {
            Ok(other) if other == parsed => ("json_request", json_request),
            _ => ("function_call_dsl", render_functions(&parsed)?),
        }
    };
    let reparsed =
        parse_request(&canonical_request).map_err(|e| format!("canonical request parse: {e}"))?;
    if reparsed != parsed {
        return Err("canonical request roundtrip changed the parsed structure".to_owned());
    }
    Ok(json!({
        "ok": true,
        "parser": PARSER,
        "parser_source_sha256": SOURCE_SHA,
        "executed": false,
        "completion": completion,
        "completion_parsed_unchanged": true,
        "mode": wire.mode,
        "ops": wire.ops,
        "ast_json_roundtrip_equal": true,
        "parser_roundtrip_equal": true,
        "canonical_format": canonical_format,
        "canonical_request": canonical_request,
    }))
}

fn validate_line(line_number: usize, line: &str) -> Value {
    let result = serde_json::from_str::<Value>(line)
        .map_err(|e| format!("input JSON: {e}"))
        .and_then(|row| {
            let completion = row
                .as_object()
                .and_then(|obj| obj.get("completion"))
                .and_then(Value::as_str)
                .ok_or_else(|| "input row must be an object with a string completion".to_owned())?;
            validate(completion)
        });
    let mut output = result.unwrap_or_else(|error| {
        json!({
            "ok": false, "parser": PARSER, "parser_source_sha256": SOURCE_SHA,
            "executed": false, "error": error,
        })
    });
    output["line"] = json!(line_number);
    output
}

fn self_test() -> Result<(), String> {
    let valid = [
        (r#"stats()"#, "single"),
        (r#"[stats(),get(id="abc")]"#, "parallel"),
        (
            r#"search(kind="entity",query="x") | get(id=$prev.items[0].id)"#,
            "chain",
        ),
        (
            r#"a() | b(data={"keys":[$prev.id,{"literal":"quotes: \"ok\", slash: \\, newline: \n, Unicode: 雪"}]},all=$prev,first=$prev[0])"#,
            "chain",
        ),
        (
            r#"create(kind="concept",name="quote: \"hi\" | comma, bracket]",properties={"x":[1,true,null,-2.5]})"#,
            "single",
        ),
        (
            r#"a() | b(id="$prev.items[0].id",literal="\\$prev.id",nested=["\\$prev.id"] )"#,
            "chain",
        ),
        (r#"a(literal="\\$prev.id")"#, "single"),
        (r#"{"tool":"stats","args":{}}"#, "single"),
        (
            r#"[{"tool":"stats"},{"tool":"get","args":{"id":"abc"}}]"#,
            "parallel",
        ),
    ];
    for (input, mode) in valid {
        let output = validate(input)?;
        if output["mode"] != mode
            || output["completion"] != input
            || output["ops"].as_array().is_none()
        {
            return Err(format!("valid fixture output mismatch: {input}"));
        }
    }
    let malformed = [
        "",
        "stats(",
        "[stats(),]",
        "a(),b()",
        "a() |",
        "get(id=$prev.id)",
        "a(x=1,x=2)",
        "a(x=bareword)",
        "a() trailing",
        "[a() | b()]",
    ];
    for input in malformed {
        if validate(input).is_ok() {
            return Err(format!("malformed fixture incorrectly accepted: {input:?}"));
        }
    }
    for line in ["", "not JSON", "{}", "[]", r#"{"completion":42}"#] {
        if validate_line(1, line)["ok"] != false {
            return Err(format!(
                "malformed JSONL fixture incorrectly accepted: {line:?}"
            ));
        }
    }
    println!(
        "{}",
        json!({"ok":true,"self_test":true,"valid":valid.len(),"malformed_dsl":malformed.len(),"malformed_jsonl":5,"parser":PARSER,"parser_source_sha256":SOURCE_SHA,"executed":false})
    );
    Ok(())
}

fn run() -> Result<bool, String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--self-test"] {
        self_test()?;
        return Ok(true);
    }
    if !args.is_empty() {
        return Err("usage: khive-dsl-validator [--self-test]; otherwise read JSONL {completion:string} on stdin".to_owned());
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    let mut all_ok = true;
    let mut count = 0;
    for (index, line) in stdin.lock().lines().enumerate() {
        count += 1;
        let row = match line {
            Ok(line) => validate_line(index + 1, &line),
            Err(error) => {
                all_ok = false;
                let row = json!({"ok":false,"line":index+1,"parser":PARSER,"executed":false,"error":format!("stdin read: {error}")});
                writeln!(output, "{row}").map_err(|e| e.to_string())?;
                break;
            }
        };
        all_ok &= row["ok"] == true;
        writeln!(output, "{row}").map_err(|e| e.to_string())?;
    }
    output.flush().map_err(|e| e.to_string())?;
    if count == 0 {
        return Err("empty input: expected at least one JSONL row".to_owned());
    }
    Ok(all_ok)
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::from(1),
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::from(2)
        }
    }
}
