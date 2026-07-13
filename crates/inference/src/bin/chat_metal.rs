//! Metal GPU generation — JSON event mode for Lattice Studio app, plus interactive REPL.
//!
//! # Usage — JSON one-shot mode (used by Lattice Studio; legacy, one process per message)
//!
//! ```
//! chat_metal --model ~/.lattice/models/qwen3.5-0.8b --prompt "Hello" --max-tokens 64 --json
//! chat_metal --model qwen3.5-0.8b-q4 --prompt "Hello" --max-tokens 64 --json
//! chat_metal --model qwen3.5-0.8b --lora adapter.safetensors --prompt "..." --json
//! ```
//!
//! # Usage — JSON serve mode (used by Lattice Studio; one process per model, keeps model warm)
//!
//! ```
//! chat_metal --model ~/.lattice/models/qwen3.5-0.8b --json --serve
//! ```
//!
//! Reads newline-delimited JSON request objects from stdin:
//! `{"prompt":"<chatml>","max_tokens":64,"temperature":0.7,"top_k":50,"top_p":0.9,"repetition_penalty":1.1,"seed":42}`
//! All fields except `prompt` are optional and fall back to the CLI defaults.
//! Emits the same `@@lattice gen_token` event stream as the one-shot path for each request.
//! Exits cleanly on stdin EOF (app closed) or broken pipe.
//!
//! Emits `@@lattice` gen_token events identical to `generate_lora` so the app parser
//! needs no changes. Every response bubble produced by this binary is labelled
//! "GPU Metal" — never CPU. (The app selects the binary; it does not label after the fact.)
//!
//! # Usage — interactive REPL (legacy; no --json)
//!
//! ```
//! cargo run --release -p lattice-inference --bin chat_metal --features "f16,metal-gpu"
//! ```
//!
//! # LoRA loading
//!
//! Loads PEFT-format or MLX-format `.safetensors` adapters inline using `SafetensorsFile`.
//! The `lattice-tune` crate is not a dependency of `lattice-inference`; we replicate the
//! minimal key-parsing logic here. Alpha defaults to `rank` (scale = 1.0), matching the
//! tune crate's own default when no `__metadata__` alpha field is present.

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("chat_metal requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("chat_metal: {e}");
            std::process::exit(1);
        }
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Escape a string as a JSON string literal (surrounding double quotes included).
/// Does NOT depend on serde_json — self-contained and correct.
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
                let code = c as u32;
                out.push_str(&format!("\\u{code:04x}"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn default_model_cache() -> std::path::PathBuf {
    std::env::var("LATTICE_MODEL_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            std::path::PathBuf::from(home)
                .join(".lattice")
                .join("models")
        })
}

// ─── LoRA loading (inline — no lattice-tune dep) ────────────────────────────

/// Read `alpha` from a safetensors file's `__metadata__` section.
/// Returns `None` when the field is absent or unparseable.
/// We parse the raw header bytes because `SafetensorsFile` skips `__metadata__`.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn read_lora_alpha_from_metadata(path: &std::path::Path) -> Option<f32> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).ok()?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes).ok()?;
    let header = std::str::from_utf8(&header_bytes).ok()?;

    // Minimal extraction: find `"__metadata__"` key, then scan for `"alpha":"<n>"`.
    let meta_pos = header.find("\"__metadata__\"")?;
    let after_meta = &header[meta_pos..];
    let brace = after_meta.find('{')?;
    let inner = &after_meta[brace + 1..];
    let end = inner.find('}')?;
    let meta_content = &inner[..end];

    // Look for "alpha":"<n>" or "alpha": "<n>" in the metadata content.
    let alpha_key = meta_content.find("\"alpha\"")?;
    let after_key = &meta_content[alpha_key + 7..];
    // Skip colon and optional whitespace, then the opening quote.
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let after_colon = after_colon.strip_prefix('"')?;
    let close_quote = after_colon.find('"')?;
    let alpha_str = &after_colon[..close_quote];
    alpha_str.parse::<f32>().ok()
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn load_lora_safetensors(
    path: &std::path::Path,
) -> Result<
    (
        Vec<lattice_inference::forward::metal_qwen35::LoraLayerData>,
        f32,
    ),
    Box<dyn std::error::Error>,
> {
    use lattice_inference::forward::metal_qwen35::LoraLayerData;
    use lattice_inference::weights::f32_weights::SafetensorsFile;

    // Read alpha from metadata before opening SafetensorsFile (which skips __metadata__).
    let metadata_alpha = read_lora_alpha_from_metadata(path);

    let sf = SafetensorsFile::open(path)?;

    // Collect all tensor names.
    let names: Vec<String> = sf
        .tensor_names()
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

    // We need to pair lora_A and lora_B tensors by layer and module.
    // Supported key formats:
    //   PEFT: base_model.model.model.layers.{i}.{self_attn|mlp}.{module}.lora_A.weight
    //         base_model.model.model.layers.{i}.{self_attn|mlp}.{module}.lora_B.weight
    //   MLX:  model.layers.{i}.{self_attn|mlp}.{module}.lora_a
    //         model.layers.{i}.{self_attn|mlp}.{module}.lora_b

    struct ParsedKey {
        layer_idx: usize,
        module: String,
        is_a: bool,
        tensor_name: String,
    }

    let mut parsed: Vec<ParsedKey> = Vec::new();
    let mut rank_global: Option<usize> = None;

    for name in &names {
        // Try PEFT format.
        if let Some(rest) = name.strip_prefix("base_model.model.model.layers.") {
            let parts: Vec<&str> = rest.splitn(6, '.').collect();
            // parts: [i, "self_attn" | "mlp", module, "lora_A" | "lora_B", "weight"]
            if parts.len() >= 4
                && let Ok(layer_idx) = parts[0].parse::<usize>()
                && matches!(parts[1], "self_attn" | "mlp")
            {
                let module = parts[2].to_string();
                let is_a = parts[3] == "lora_A";
                let is_b = parts[3] == "lora_B";
                if is_a || is_b {
                    parsed.push(ParsedKey {
                        layer_idx,
                        module,
                        is_a,
                        tensor_name: name.clone(),
                    });
                }
            }
            continue;
        }

        // Try MLX format.
        if let Some(rest) = name.strip_prefix("model.layers.") {
            let parts: Vec<&str> = rest.splitn(5, '.').collect();
            // parts: [i, "self_attn" | "mlp", module, "lora_a" | "lora_b"]
            if parts.len() >= 4
                && let Ok(layer_idx) = parts[0].parse::<usize>()
                && matches!(parts[1], "self_attn" | "mlp")
            {
                let module = parts[2].to_string();
                let is_a = parts[3] == "lora_a";
                let is_b = parts[3] == "lora_b";
                if is_a || is_b {
                    parsed.push(ParsedKey {
                        layer_idx,
                        module,
                        is_a,
                        tensor_name: name.clone(),
                    });
                }
            }
        }
    }

    if parsed.is_empty() {
        return Err(format!(
            "no LoRA tensors found in {path:?} (checked PEFT and MLX key formats)"
        )
        .into());
    }

    // Group by (layer_idx, module) and build LoraLayerData entries.
    let mut groups: std::collections::HashMap<(usize, String), (Option<String>, Option<String>)> =
        std::collections::HashMap::new();

    for pk in &parsed {
        let entry = groups
            .entry((pk.layer_idx, pk.module.clone()))
            .or_insert((None, None));
        if pk.is_a {
            entry.0 = Some(pk.tensor_name.clone());
        } else {
            entry.1 = Some(pk.tensor_name.clone());
        }
    }

    if groups
        .values()
        .all(|(a_name, b_name)| a_name.is_none() || b_name.is_none())
    {
        return Err(
            "LoRA loader: all tensor pairs were incomplete (missing A or B for every group)".into(),
        );
    }

    let mut layers: Vec<LoraLayerData> = Vec::new();

    for ((layer_idx, module), (a_name, b_name)) in &groups {
        let (a_name, b_name) = match (a_name, b_name) {
            (Some(a_name), Some(b_name)) => (a_name, b_name),
            (Some(a_name), None) => {
                return Err(
                    format!("LoRA adapter has tensor '{a_name}' but no matching B tensor").into(),
                );
            }
            (None, Some(b_name)) => {
                return Err(
                    format!("LoRA adapter has tensor '{b_name}' but no matching A tensor").into(),
                );
            }
            (None, None) => {
                return Err(format!(
                    "LoRA adapter has an empty tensor group at layer {layer_idx} module '{module}'"
                )
                .into());
            }
        };

        let (a_data, a_shape) = sf.get_f32_tensor(a_name)?;
        let (b_data, b_shape) = sf.get_f32_tensor(b_name)?;

        // PEFT format: A is (rank, d_in), B is (d_out, rank).
        // MLX format: A is (d_in, rank), B is (rank, d_out) — must transpose both.
        //
        // Determine format by checking which tensor name was matched.
        // MLX keys are "lora_a"/"lora_b" (lowercase, no .weight suffix).
        let is_mlx = a_name.ends_with(".lora_a") || a_name.ends_with(".lora_b");

        let (a_vec, rank, d_in, b_vec, d_out) = if is_mlx {
            // MLX A: (d_in, rank) → transpose to (rank, d_in)
            if a_shape.len() != 2 || b_shape.len() != 2 {
                return Err(format!(
                    "unexpected shape for MLX LoRA tensors at layer {layer_idx} module {module}"
                )
                .into());
            }
            let a_d_in = a_shape[0];
            let a_rank = a_shape[1];
            let b_rank = b_shape[0];
            let b_d_out = b_shape[1];
            if a_rank != b_rank {
                return Err(format!(
                    "rank mismatch at layer {layer_idx} module {module}: A rank={a_rank} B rank={b_rank}"
                )
                .into());
            }
            // Transpose A: (d_in, rank) → (rank, d_in)
            let mut a_t = vec![0.0f32; a_d_in * a_rank];
            for r in 0..a_rank {
                for d in 0..a_d_in {
                    a_t[r * a_d_in + d] = a_data[d * a_rank + r];
                }
            }
            // Transpose B: (rank, d_out) → (d_out, rank)
            let mut b_t = vec![0.0f32; b_d_out * b_rank];
            for r in 0..b_rank {
                for d in 0..b_d_out {
                    b_t[d * b_rank + r] = b_data[r * b_d_out + d];
                }
            }
            (a_t, a_rank, a_d_in, b_t, b_d_out)
        } else {
            // PEFT A: (rank, d_in), B: (d_out, rank) — already correct layout.
            if a_shape.len() != 2 || b_shape.len() != 2 {
                return Err(format!(
                    "unexpected shape for PEFT LoRA tensors at layer {layer_idx} module {module}"
                )
                .into());
            }
            let rank = a_shape[0];
            let d_in = a_shape[1];
            let d_out = b_shape[0];
            let b_rank_check = b_shape[1];
            if b_rank_check != rank {
                return Err(format!(
                    "rank mismatch at layer {layer_idx} module {module}: A rank={rank} B rank={b_rank_check}"
                )
                .into());
            }
            (a_data.to_vec(), rank, d_in, b_data.to_vec(), d_out)
        };

        match rank_global {
            None => rank_global = Some(rank),
            Some(prev) if prev != rank => {
                return Err(format!(
                    "inconsistent LoRA ranks: first seen rank={prev}, layer {layer_idx} module '{module}' has rank={rank}"
                )
                .into());
            }
            _ => {}
        }

        layers.push(LoraLayerData {
            layer_idx: *layer_idx,
            module: module.clone(),
            a: a_vec,
            b: b_vec,
            rank,
            d_in,
            d_out,
        });
    }

    if layers.is_empty() {
        return Err(
            "LoRA loader: all tensor pairs were incomplete (missing A or B for every group)".into(),
        );
    }

    // Sort by layer_idx for deterministic load order.
    layers.sort_by_key(|l| (l.layer_idx, l.module.clone()));

    // Compute scale = alpha / rank.
    // Prefer __metadata__.alpha when available; fall back to alpha = rank (scale = 1.0),
    // matching the tune crate's own default for adapters without embedded metadata.
    let rank = rank_global.unwrap_or(1);
    let alpha = metadata_alpha.unwrap_or(rank as f32);
    let scale = if rank > 0 { alpha / rank as f32 } else { 1.0 };

    eprintln!(
        "[chat_metal] LoRA: {} layer×module pairs, rank={}, alpha={}, scale={:.2}",
        layers.len(),
        rank,
        alpha,
        scale
    );

    Ok((layers, scale))
}

// ─── JSON generation helper (shared by one-shot and serve paths) ────────────

/// Emit a single streaming JSON generation response on stdout.
///
/// Writes one `@@lattice gen_token` line per token (done:false), then a final
/// done:true line with tok_s and ttft_ms. Format is byte-identical to generate_lora
/// so the app parser needs no changes. Returns Err on broken pipe or flush failure.
///
/// Generation itself runs through the cross-turn prefix cache (#462): a
/// same-slot conversation that safely extends the previous call reuses its KV
/// state instead of re-prefilling from scratch. A rare cache/generation-level
/// failure (already fail-closed and cleaned up inside the engine) is surfaced
/// as an `{"ev":"error","msg":...}` line and this function still returns `Ok`,
/// so one bad request doesn't tear down an otherwise-healthy serve loop.
///
/// Used by both `--json --prompt` (one-shot) and `--json --serve` (serve loop) so
/// the event format cannot drift between the two paths.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn emit_json_generation(
    prompt: &str,
    metal: &mut lattice_inference::forward::metal_qwen35::MetalQwen35State,
    tokenizer: &lattice_inference::tokenizer::bpe::BpeTokenizer,
    gen_cfg: &lattice_inference::model::qwen35_config::GenerateConfig,
    model_format: &str,
    lora_tag: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::kv_cache::CrossTurnSlotId;
    use std::io::Write;

    let mut stdout = std::io::stdout();
    let mut first_token_emitted = false;
    let mut ttft_ms: f64 = 0.0;
    let t1 = std::time::Instant::now();

    // Path-proof probe (issue #239): opt-in so normal Lattice Studio runs pay no
    // cost. When enabled, zero the dispatch-site counters before this generation
    // so the marker below reflects only this call's Metal attention/KV dispatches.
    let path_proof_enabled = matches!(
        std::env::var("LATTICE_METAL_PATH_PROOF").as_deref(),
        Ok("1") | Ok("true")
    );
    if path_proof_enabled {
        metal.reset_path_proof_counters();
    }

    // Capture the first write/flush failure so we can stop generation and return Err
    // rather than silently completing a run whose output was never received.
    let mut stream_err: Option<std::io::Error> = None;
    // Cache-aware call (#462): reuses the previous request's shared token prefix
    // instead of the caller reset_state()-ing before every request. Safe even when
    // unrelated conversations are interleaved through one warm serve process — a
    // divergent prompt just plans a full refill internally, matching the old
    // unconditional-reset behavior exactly (see kv_cache::cross_turn).
    let cache_result = metal.generate_streaming_with_prefix_cache(
        CrossTurnSlotId::DEFAULT,
        prompt,
        tokenizer,
        gen_cfg,
        |delta, token_id| {
            if !first_token_emitted {
                ttft_ms = t1.elapsed().as_secs_f64() * 1000.0;
                first_token_emitted = true;
            }
            let token_json = json_escape(delta);
            let write_result = writeln!(
                stdout,
                "@@lattice {{\"ev\":\"gen_token\",\"token\":{token_json},\"token_id\":{token_id},\"done\":false}}"
            )
            .and_then(|()| stdout.flush());
            if let Err(e) = write_result {
                stream_err = Some(e);
                return false; // downstream consumer is gone — stop generating
            }
            true // continue generation
        },
    );

    if let Some(e) = stream_err {
        return Err(format!("streaming stdout write failed: {e}").into());
    }

    // A cache/generation failure is fail-closed at the engine level already (live
    // KV/GDN state and the retained prefix entry are both reset before the error
    // is returned), so the warm process is left in a clean state. Surface it as a
    // protocol-level error event instead of tearing down the whole serve loop over
    // one bad request.
    let cached = match cache_result {
        Ok(cached) => cached,
        Err(e) => {
            let msg = json_escape(&format!("generation failed: {e}"));
            writeln!(stdout, "@@lattice {{\"ev\":\"error\",\"msg\":{msg}}}")
                .and_then(|()| stdout.flush())
                .map_err(|e| format!("error-event write failed: {e}"))?;
            return Ok(());
        }
    };
    let output = cached.output;

    let gen_ms = t1.elapsed().as_millis();
    let tok_s = if gen_ms > 0 {
        output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
    } else {
        0.0
    };

    // Final done event — format identical to generate_lora so app parser needs no change.
    // prompt_tokens / gen_tokens / total_ms are additive: older parsers ignore unknown fields.
    writeln!(
        stdout,
        "@@lattice {{\"ev\":\"gen_token\",\"token\":\"\",\"done\":true,\"tok_s\":{tok_s:.1},\"ttft_ms\":{ttft_ms:.1},\"prompt_tokens\":{},\"gen_tokens\":{},\"total_ms\":{}}}",
        output.prompt_tokens, output.generated_tokens, gen_ms
    )
    .and_then(|()| stdout.flush())
    .map_err(|e| format!("streaming done-event write failed: {e}"))?;

    eprintln!(
        "[chat_metal] GPU Metal {model_format}{lora_tag}: {} prompt + {} gen in {}ms = {tok_s:.1} tok/s | cache: {:?} reused {}/{}",
        output.prompt_tokens,
        output.generated_tokens,
        gen_ms,
        cached.cache.mode,
        cached.cache.reused_tokens,
        cached.cache.prompt_tokens,
    );

    // Emit the runtime path-proof marker (issue #239): CI gates on this to prove
    // the required Metal attention/KV dispatches actually ran, rather than
    // trusting a "Metal device present" check that a paravirtual CI runner would
    // pass vacuously. See scripts/e2e_parity_check.py for the fail-closed parser.
    if path_proof_enabled {
        let s = metal.path_proof_snapshot();
        eprintln!(
            "[METAL_PATH_PROOF] prefill_kv_batch={} prefill_attn_batched={} decode_kv_copy={} decode_attn_direct={} decode_attn_split_partial={} decode_attn_split_reduce={} kv_f16={}",
            s.prefill_kv_batch,
            s.prefill_attn_batched,
            s.decode_kv_copy,
            s.decode_attn_direct,
            s.decode_attn_split_partial,
            s.decode_attn_split_reduce,
            s.kv_f16,
        );
    }

    Ok(())
}

// ─── --json --serve request parsing (#654) ─────────────────────────────────
//
// Factored out of the serve loop in `run()` below so the capability-matrix
// fixtures (`mod tests`) can exercise the exact wire-parsing contract for
// `--json --serve` request lines without a loaded Metal model. Pure — no
// Metal/GPU dependency at all — so it is not behind the `metal-gpu` cfg gate
// the rest of this binary uses; it is still only reachable from `run()`,
// which is gated. No behavior change versus the inline code it replaces.

/// CLI-level defaults a per-request field falls back to when the incoming
/// JSON line omits it. Captured once from the parsed `--*` flags in `run()`.
#[derive(Clone, Copy)]
struct ServeRequestDefaults {
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    reasoning_budget: Option<usize>,
}

/// One request parsed from a newline-delimited JSON line in `--json --serve`
/// mode. Every field but `prompt` falls back to `ServeRequestDefaults` when
/// absent, mirroring the doc comment's advertised wire shape:
/// `{"prompt":"<chatml>","max_tokens":64,"temperature":0.7,"top_k":50,"top_p":0.9,"repetition_penalty":1.1,"seed":42}`.
#[derive(Debug)]
struct ParsedServeRequest {
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    reasoning_budget: Option<usize>,
}

/// Why a `--json --serve` request line could not be turned into a
/// `ParsedServeRequest`. Both variants map 1:1 to the `@@lattice`
/// `{"ev":"error",...}` line `run()`'s serve loop emits on this `Err`.
#[derive(Debug)]
enum ServeRequestError {
    /// The line was not valid JSON at all. Carries the raw `serde_json`
    /// parse error message (embedded in the emitted error event, same as
    /// before this factoring).
    Malformed(String),
    /// The line parsed as JSON but had no string `"prompt"` field.
    MissingPrompt,
}

fn parse_serve_request_line(
    line: &str,
    defaults: ServeRequestDefaults,
) -> Result<ParsedServeRequest, ServeRequestError> {
    let req_val: serde_json::Value =
        serde_json::from_str(line).map_err(|e| ServeRequestError::Malformed(e.to_string()))?;

    let prompt = req_val["prompt"]
        .as_str()
        .ok_or(ServeRequestError::MissingPrompt)?
        .to_owned();

    let max_tokens = req_val["max_tokens"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(defaults.max_tokens);
    let temperature = req_val["temperature"]
        .as_f64()
        .map(|v| v as f32)
        .unwrap_or(defaults.temperature);
    let top_k = req_val["top_k"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(defaults.top_k);
    let top_p = req_val["top_p"]
        .as_f64()
        .map(|v| v as f32)
        .unwrap_or(defaults.top_p);
    let repetition_penalty = req_val["repetition_penalty"]
        .as_f64()
        .map(|v| v as f32)
        .unwrap_or(defaults.repetition_penalty);
    let seed = req_val["seed"].as_u64().or(defaults.seed);
    let reasoning_budget = req_val["reasoning_budget"]
        .as_u64()
        .map(|v| v as usize)
        .filter(|&n| n > 0)
        .or(defaults.reasoning_budget);

    Ok(ParsedServeRequest {
        prompt,
        max_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        reasoning_budget,
    })
}

// ─── main logic ─────────────────────────────────────────────────────────────

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
    use lattice_inference::kv_cache::CrossTurnSlotId;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{
        GenerateConfig, QWEN_CHAT_IM_END_TOKEN_ID, Qwen35Config,
    };
    use lattice_inference::model_format::{self, ModelFormat};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use std::io::Write;

    let args: Vec<String> = std::env::args().collect();

    // ── Arg parsing ──────────────────────────────────────────────────────────

    let emit_json = parse_flag(&args, "--json");
    let serve_mode = parse_flag(&args, "--serve");

    let prompt_opt = parse_arg(&args, "--prompt");

    let max_tokens: usize = parse_arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let seed: Option<u64> = parse_arg(&args, "--seed").and_then(|s| s.parse().ok());

    let temperature: f32 = parse_arg(&args, "--temperature")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);

    // Sampler knobs. Defaults match GenerateConfig::default() (50 / 0.9 / 1.1),
    // so omitting these flags is byte-identical to the prior hardcoded behavior.
    let top_k: usize = parse_arg(&args, "--top-k")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let top_p: f32 = parse_arg(&args, "--top-p")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9);

    let repetition_penalty: f32 = parse_arg(&args, "--repetition-penalty")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.1);

    let reasoning_budget: Option<usize> = parse_arg(&args, "--reasoning-budget")
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0);

    let lora_path: Option<std::path::PathBuf> = parse_arg(&args, "--lora").map(|s| {
        let p = std::path::PathBuf::from(&s);
        if p.is_absolute() {
            p
        } else {
            std::env::current_dir().unwrap_or_default().join(p)
        }
    });

    // --model <dir> or --model <name> (resolved relative to model cache)
    // Legacy env var fallback for interactive REPL.
    let model_dir = if let Some(dir) = parse_arg(&args, "--model-dir") {
        std::path::PathBuf::from(dir)
    } else if let Some(model_name_or_path) = parse_arg(&args, "--model") {
        let p = std::path::PathBuf::from(&model_name_or_path);
        if p.is_absolute() || p.starts_with("~") {
            p
        } else if p.components().count() == 1 {
            // Just a model name like "qwen3.5-0.8b" — resolve from cache.
            default_model_cache().join(model_name_or_path)
        } else {
            p
        }
    } else {
        // Legacy env var path for backward-compatible REPL usage.
        let home = std::env::var("HOME")?;
        let dir_str = std::env::var("LATTICE_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-2b"));
        std::path::PathBuf::from(dir_str)
    };

    // ── Detect model format ──────────────────────────────────────────────────

    let dir = model_dir.as_path();

    let format = model_format::detect_format(dir);

    // Tokenizer lives alongside the model; for Q4 models --tokenizer-dir or env var override.
    let tokenizer_dir_str = parse_arg(&args, "--tokenizer-dir")
        .or_else(|| parse_arg(&args, "--tokenizer"))
        .or_else(|| std::env::var("LATTICE_TOKENIZER_DIR").ok())
        .unwrap_or_else(|| dir.to_string_lossy().into());
    let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);

    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))
        .map_err(|e| {
            format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_dir.display()
            )
        })?;

    // ── Load model ───────────────────────────────────────────────────────────

    let mut metal;
    let model_format_label; // "bf16" | "q4"

    match format {
        ModelFormat::Q4 => {
            eprintln!(
                "[chat_metal] Detected Q4 model directory: {}",
                dir.display()
            );
            let cfg = if dir.join("config.json").exists() {
                Qwen35Config::from_config_json(&dir.join("config.json"))
                    .map_err(|e| format!("failed to parse config.json: {e}"))?
            } else {
                eprintln!("[chat_metal] No config.json; using qwen36_27b preset");
                Qwen35Config::qwen36_27b()
            };
            eprintln!("[chat_metal] Loading Q4 model...");
            let t0 = std::time::Instant::now();
            metal = MetalQwen35State::from_q4_dir(
                dir,
                &tokenizer_dir.join("tokenizer.json"),
                &cfg,
                4096,
            )
            .map_err(|e| format!("failed to initialize Metal from Q4 dir: {e}"))?;
            eprintln!(
                "[chat_metal] Q4 model loaded in {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            model_format_label = "q4";
        }
        ModelFormat::Safetensors => {
            eprintln!("[chat_metal] Loading bf16 model from {}...", dir.display());
            let t0 = std::time::Instant::now();
            let model = Qwen35Model::from_safetensors(dir)
                .map_err(|e| format!("failed to load model: {e}"))?;
            let cfg = model.config().clone();
            eprintln!(
                "[chat_metal] Model loaded in {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            eprintln!("[chat_metal] Initializing Metal GPU...");
            let t1 = std::time::Instant::now();
            metal = MetalQwen35State::new(model.weights(), &cfg, 4096)
                .map_err(|e| format!("failed to init Metal: {e}"))?;
            eprintln!(
                "[chat_metal] Metal ready in {:.1}s",
                t1.elapsed().as_secs_f64()
            );
            model_format_label = "bf16";
        }
        ModelFormat::Unknown => {
            return Err(model_format::unrecognized_format_message(dir).into());
        }
    }

    // ── Load LoRA adapter (if requested) ────────────────────────────────────

    let has_lora = lora_path.is_some();
    if let Some(ref lp) = lora_path {
        eprintln!("[chat_metal] Loading LoRA adapter from {}...", lp.display());
        let t_lora = std::time::Instant::now();
        let (layers, scale) = load_lora_safetensors(lp)?;
        metal
            .load_lora_adapter(layers, scale, None)
            .map_err(|e| format!("load_lora_adapter failed: {e}"))?;
        eprintln!(
            "[chat_metal] LoRA loaded in {:.1}s",
            t_lora.elapsed().as_secs_f64()
        );
    }

    // ── GenerateConfig ───────────────────────────────────────────────────────

    let gen_cfg = GenerateConfig {
        max_new_tokens: max_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
        enable_thinking: true,
        enable_mtp: None,
        grammar: None,
        stop_strings: vec![],
        reasoning_budget,
        logprobs: None,
    };

    // ── JSON modes (used by Lattice Studio app) ─────────────────────────────

    if emit_json {
        let lora_tag = if has_lora { "+lora" } else { "" };

        if serve_mode {
            // Persistent serve loop: load once, process requests until stdin closes.
            // Each request is a newline-delimited JSON object; responses are the same
            // @@lattice gen_token event stream as the one-shot path.
            use std::io::BufRead;
            let stdin = std::io::stdin();
            let stdin_lock = stdin.lock();
            let mut lines = stdin_lock.lines();

            let serve_defaults = ServeRequestDefaults {
                max_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
                reasoning_budget,
            };

            // The model is fully loaded by this point (built above, before the serve loop).
            // Emit a `ready` event so the app can spawn this process from a "Load model" button
            // and show an honest LOADED state once the weights are resident — without having to
            // send a generation request first.
            {
                use std::io::Write;
                let mut out = std::io::stdout();
                let _ = writeln!(out, "@@lattice {{\"ev\":\"ready\"}}");
                let _ = out.flush();
            }

            loop {
                let line = match lines.next() {
                    None => break, // stdin closed (app teardown) → exit cleanly
                    Some(Ok(l)) => l,
                    Some(Err(_)) => break, // read error → exit cleanly
                };
                let line = line.trim().to_owned();
                if line.is_empty() {
                    continue;
                }

                // Parse the request line (#654: factored into a pure function,
                // `parse_serve_request_line` below, so the capability-matrix
                // fixtures can exercise this exact wire contract without a
                // loaded Metal model). Same two failure modes, same messages.
                let parsed = match parse_serve_request_line(&line, serve_defaults) {
                    Ok(p) => p,
                    Err(ServeRequestError::Malformed(e)) => {
                        let mut out = std::io::stdout();
                        use std::io::Write;
                        let msg = json_escape(&format!("malformed request: {e}"));
                        let _ = writeln!(out, "@@lattice {{\"ev\":\"error\",\"msg\":{msg}}}");
                        let _ = out.flush();
                        continue; // robustness: keep loop alive on bad input
                    }
                    Err(ServeRequestError::MissingPrompt) => {
                        let mut out = std::io::stdout();
                        use std::io::Write;
                        let _ = writeln!(
                            out,
                            "@@lattice {{\"ev\":\"error\",\"msg\":\"missing \\\"prompt\\\" field\"}}"
                        );
                        let _ = out.flush();
                        continue;
                    }
                };
                let prompt = parsed.prompt;

                let req_cfg = GenerateConfig {
                    max_new_tokens: parsed.max_tokens,
                    temperature: parsed.temperature,
                    top_k: parsed.top_k,
                    top_p: parsed.top_p,
                    repetition_penalty: parsed.repetition_penalty,
                    seed: parsed.seed,
                    stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
                    enable_thinking: true,
                    enable_mtp: None,
                    grammar: None,
                    stop_strings: vec![],
                    reasoning_budget: parsed.reasoning_budget,
                    logprobs: None,
                };

                // Each request's full ChatML history is re-sent in `prompt` (the
                // client is stateless), but the engine itself now reuses the KV
                // state shared with the previous request when it's a safe append
                // instead of unconditionally reset_state()-ing here (#462). A
                // first request, a divergent conversation, or an edited history
                // all just fall back to a full refill inside emit_json_generation.

                // Broken pipe means the app closed its read end — stop the loop cleanly.
                if let Err(e) = emit_json_generation(
                    &prompt,
                    &mut metal,
                    &tokenizer,
                    &req_cfg,
                    model_format_label,
                    lora_tag,
                ) {
                    eprintln!("[chat_metal] serve: generation stopped: {e}");
                    break;
                }
            }

            return Ok(());
        }

        // One-shot mode (--json without --serve): requires --prompt.
        let Some(prompt) = prompt_opt else {
            return Err(
                "--json mode requires --prompt (or --serve for persistent serve mode)".into(),
            );
        };
        emit_json_generation(
            &prompt,
            &mut metal,
            &tokenizer,
            &gen_cfg,
            model_format_label,
            lora_tag,
        )?;
        return Ok(());
    }

    // ── Interactive REPL mode (no --json) ───────────────────────────────────

    use std::io::BufRead;

    let lora_tag = if has_lora { "+LoRA" } else { "" };
    eprintln!("\n=== GPU Metal {model_format_label}{lora_tag} — Qwen3.5 Chat ===");
    eprintln!("Type your message. Empty line or Ctrl-D to quit.\n");

    let system_msg = ChatMessage::system("You are a helpful assistant. Be concise and direct.");
    let stdin = std::io::stdin();
    let mut history: Vec<ChatMessage> = vec![system_msg];

    loop {
        print!("> ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }

        let input = line.trim();
        if input.is_empty() {
            break;
        }

        // In REPL mode we support single --prompt as a one-shot and then enter the loop.
        // Build a chat prompt from history.
        history.push(ChatMessage::user(input));

        let t = std::time::Instant::now();

        // Use the cache-aware entry point so turn N reuses turn N-1's shared
        // prefix instead of unconditionally reset_state()-ing and re-prefilling
        // the whole conversation every turn (#462). Falls back to a full prefill
        // on its own whenever the shared prefix isn't a safe append — new
        // session, edited history, adapter change, etc. — see kv_cache::cross_turn.
        let mut response_text = String::new();
        let cache_result = metal.chat_completion_streaming_with_prefix_cache(
            CrossTurnSlotId::DEFAULT,
            &history,
            &tokenizer,
            &gen_cfg,
            |delta, _| {
                print!("{delta}");
                std::io::stdout().flush().ok();
                response_text.push_str(delta);
                true
            },
        );
        println!();

        let cached = match cache_result {
            Ok(cached) => cached,
            Err(e) => {
                eprintln!("[chat_metal] generation failed: {e}");
                // The engine already reset its live state on this error path, so
                // just drop the unanswered turn and let the user retry cleanly.
                history.pop();
                continue;
            }
        };
        let result = cached.output;

        let elapsed = t.elapsed();
        let tps = if result.completion_tokens > 0 {
            result.completion_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "[{} prompt + {} gen in {:.1}ms = {:.1} tok/s | GPU Metal {model_format_label} | cache: {:?} reused {}/{}]",
            result.prompt_tokens,
            result.completion_tokens,
            elapsed.as_secs_f64() * 1000.0,
            tps,
            cached.cache.mode,
            cached.cache.reused_tokens,
            cached.cache.prompt_tokens,
        );

        history.push(ChatMessage::assistant(response_text.trim()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_lora_safetensors(path: &std::path::Path, tensors: &[(&str, Vec<usize>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();

        for (name, shape) in tensors {
            let start = data.len();
            let byte_len = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
            data.resize(start + byte_len, 0);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, start + byte_len],
                }),
            );
        }

        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn lora_loader_accepts_attention_and_mlp_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let fixtures = [
            (
                "peft.safetensors",
                vec![
                    (
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                        vec![2, 4],
                    ),
                    (
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                        vec![6, 2],
                    ),
                    (
                        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
                        vec![2, 4],
                    ),
                    (
                        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight",
                        vec![8, 2],
                    ),
                ],
            ),
            (
                "mlx.safetensors",
                vec![
                    ("model.layers.0.self_attn.q_proj.lora_a", vec![4, 2]),
                    ("model.layers.0.self_attn.q_proj.lora_b", vec![2, 6]),
                    ("model.layers.0.mlp.gate_proj.lora_a", vec![4, 2]),
                    ("model.layers.0.mlp.gate_proj.lora_b", vec![2, 8]),
                ],
            ),
        ];

        for (filename, tensors) in fixtures {
            let path = dir.path().join(filename);
            write_lora_safetensors(&path, &tensors);

            let (layers, _) = load_lora_safetensors(&path).unwrap();
            let modules: Vec<&str> = layers.iter().map(|layer| layer.module.as_str()).collect();
            assert_eq!(modules, ["gate_proj", "q_proj"]);
        }
    }

    #[test]
    fn lora_loader_rejects_missing_b_with_valid_pair_present() {
        const ORPHAN_A: &str = "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing_b.safetensors");
        write_lora_safetensors(
            &path,
            &[
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![2, 4],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![6, 2],
                ),
                (ORPHAN_A, vec![2, 4]),
            ],
        );

        let Err(err) = load_lora_safetensors(&path) else {
            panic!("adapter with an orphan A tensor must be rejected");
        };
        assert!(err.to_string().contains(ORPHAN_A));
    }

    #[test]
    fn lora_loader_rejects_mixed_ranks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed_rank.safetensors");
        write_lora_safetensors(
            &path,
            &[
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![2, 4],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![6, 2],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight",
                    vec![3, 4],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight",
                    vec![8, 3],
                ),
            ],
        );

        let Err(err) = load_lora_safetensors(&path) else {
            panic!("adapter with mixed ranks must be rejected");
        };
        assert!(err.to_string().contains("inconsistent LoRA ranks"));
    }

    fn defaults() -> ServeRequestDefaults {
        ServeRequestDefaults {
            max_tokens: 64,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: None,
            reasoning_budget: None,
        }
    }

    // -----------------------------------------------------------------------
    // Capability-matrix fixtures (#654). Fixture IDs cited from
    // `docs/capability-matrix.md`'s Fixture manifest section;
    // `scripts/check-capability-matrix.sh` greps this file for
    // `fn <fixture_id>` and fails the build if a matrix row cites an ID that
    // no longer exists here.
    // -----------------------------------------------------------------------

    #[test]
    fn cm_chat_metal_serve_prompt_only_uses_all_defaults() {
        let parsed = parse_serve_request_line(r#"{"prompt":"hi"}"#, defaults()).unwrap();
        assert_eq!(parsed.prompt, "hi");
        assert_eq!(parsed.max_tokens, 64);
        assert_eq!(parsed.temperature, 0.7);
        assert_eq!(parsed.top_k, 50);
        assert_eq!(parsed.top_p, 0.9);
        assert_eq!(parsed.repetition_penalty, 1.1);
        assert_eq!(parsed.seed, None);
        assert_eq!(parsed.reasoning_budget, None);
    }

    #[test]
    fn cm_chat_metal_serve_per_request_overrides_all_fields() {
        let line = r#"{"prompt":"hi","max_tokens":128,"temperature":0.2,"top_k":10,
            "top_p":0.5,"repetition_penalty":1.3,"seed":7,"reasoning_budget":256}"#;
        let parsed = parse_serve_request_line(line, defaults()).unwrap();
        assert_eq!(parsed.max_tokens, 128);
        assert_eq!(parsed.temperature, 0.2);
        assert_eq!(parsed.top_k, 10);
        assert_eq!(parsed.top_p, 0.5);
        assert_eq!(parsed.repetition_penalty, 1.3);
        assert_eq!(parsed.seed, Some(7));
        assert_eq!(parsed.reasoning_budget, Some(256));
    }

    #[test]
    fn cm_chat_metal_serve_malformed_json_rejected() {
        let err = parse_serve_request_line("not json", defaults()).unwrap_err();
        assert!(matches!(err, ServeRequestError::Malformed(_)));
    }

    #[test]
    fn cm_chat_metal_serve_missing_prompt_rejected() {
        let err = parse_serve_request_line(r#"{"max_tokens":64}"#, defaults()).unwrap_err();
        assert!(matches!(err, ServeRequestError::MissingPrompt));
    }

    #[test]
    fn cm_chat_metal_serve_zero_reasoning_budget_falls_back_to_default() {
        // Matches the inline `.filter(|&n| n > 0)` this replaces: a
        // request-supplied `reasoning_budget: 0` is treated as absent, not
        // as an explicit zero-budget override.
        let d = ServeRequestDefaults {
            reasoning_budget: Some(512),
            ..defaults()
        };
        let parsed =
            parse_serve_request_line(r#"{"prompt":"hi","reasoning_budget":0}"#, d).unwrap();
        assert_eq!(parsed.reasoning_budget, Some(512));
    }

    #[test]
    fn cm_chat_metal_serve_stateless_no_stop_strings_or_history_fields() {
        // Capability-matrix row: unlike the two HTTP surfaces, chat_metal
        // --json --serve has no `stop`/`messages` wire fields at all -- the
        // client resends the full ChatML-rendered history as `prompt` on
        // every request (see the module doc comment's wire shape). This
        // fixture pins that: `stop` in the request line is inert, not an
        // error and not applied, because `ParsedServeRequest` has no field
        // for it.
        let parsed =
            parse_serve_request_line(r#"{"prompt":"hi","stop":["\n"]}"#, defaults()).unwrap();
        assert_eq!(parsed.prompt, "hi");
    }
}
