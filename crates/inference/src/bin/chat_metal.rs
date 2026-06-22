/// Metal GPU generation — JSON event mode for Lattice Studio app, plus interactive REPL.
///
/// # Usage — JSON streaming mode (used by Lattice Studio)
///
/// ```
/// chat_metal --model ~/.lattice/models/qwen3.5-0.8b --prompt "Hello" --max-tokens 64 --json
/// chat_metal --model qwen3.5-0.8b-q4 --prompt "Hello" --max-tokens 64 --json
/// chat_metal --model qwen3.5-0.8b --lora adapter.safetensors --prompt "..." --json
/// ```
///
/// Emits `@@lattice` gen_token events identical to `generate_lora` so the app parser
/// needs no changes. Every response bubble produced by this binary is labelled
/// "GPU Metal" — never CPU. (The app selects the binary; it does not label after the fact.)
///
/// # Usage — interactive REPL (legacy; no --json)
///
/// ```
/// cargo run --release -p lattice-inference --bin chat_metal --features "f16,metal-gpu"
/// ```
///
/// # LoRA loading
///
/// Loads PEFT-format or MLX-format `.safetensors` adapters inline using `SafetensorsFile`.
/// The `lattice-tune` crate is not a dependency of `lattice-inference`; we replicate the
/// minimal key-parsing logic here. Alpha defaults to `rank` (scale = 1.0), matching the
/// tune crate's own default when no `__metadata__` alpha field is present.

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
    let names: Vec<String> = sf.tensor_names().iter().map(|s| s.to_string()).collect();

    // We need to pair lora_A and lora_B tensors by layer and module.
    // Supported key formats:
    //   PEFT: base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight
    //         base_model.model.model.layers.{i}.self_attn.{module}.lora_B.weight
    //   MLX:  model.layers.{i}.self_attn.{module}.lora_a
    //         model.layers.{i}.self_attn.{module}.lora_b

    struct ParsedKey {
        layer_idx: usize,
        module: String,
        is_a: bool,
        tensor_name: String,
    }

    let mut parsed: Vec<ParsedKey> = Vec::new();
    let mut rank_global: Option<usize> = None;

    for name in &names {
        // Try PEFT format: base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight
        if let Some(rest) = name.strip_prefix("base_model.model.model.layers.") {
            let parts: Vec<&str> = rest.splitn(6, '.').collect();
            // parts: [i, "self_attn", module, "lora_A" | "lora_B", "weight"]
            if parts.len() >= 4 {
                if let Ok(layer_idx) = parts[0].parse::<usize>() {
                    if parts[1] == "self_attn" {
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
                }
            }
            continue;
        }

        // Try MLX format: model.layers.{i}.self_attn.{module}.lora_a / lora_b
        if let Some(rest) = name.strip_prefix("model.layers.") {
            let parts: Vec<&str> = rest.splitn(5, '.').collect();
            // parts: [i, "self_attn", module, "lora_a" | "lora_b"]
            if parts.len() >= 4 {
                if let Ok(layer_idx) = parts[0].parse::<usize>() {
                    if parts[1] == "self_attn" {
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
        }
    }

    if parsed.is_empty() {
        return Err(format!(
            "no LoRA tensors found in {:?} (checked PEFT and MLX key formats)",
            path
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

    let mut layers: Vec<LoraLayerData> = Vec::new();

    for ((layer_idx, module), (a_name, b_name)) in &groups {
        let (a_name, b_name) = match (a_name, b_name) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                eprintln!(
                    "[chat_metal] skipping layer {} module {}: missing A or B tensor",
                    layer_idx, module
                );
                continue;
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
                    "unexpected shape for MLX LoRA tensors at layer {} module {}",
                    layer_idx, module
                )
                .into());
            }
            let a_d_in = a_shape[0];
            let a_rank = a_shape[1];
            let b_rank = b_shape[0];
            let b_d_out = b_shape[1];
            if a_rank != b_rank {
                return Err(format!(
                    "rank mismatch at layer {} module {}: A rank={} B rank={}",
                    layer_idx, module, a_rank, b_rank
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
                    "unexpected shape for PEFT LoRA tensors at layer {} module {}",
                    layer_idx, module
                )
                .into());
            }
            let rank = a_shape[0];
            let d_in = a_shape[1];
            let d_out = b_shape[0];
            if b_shape[1] != rank {
                return Err(format!(
                    "rank mismatch at layer {} module {}: A rank={} B rank={}",
                    layer_idx, module, rank, b_shape[1]
                )
                .into());
            }
            (a_data.to_vec(), rank, d_in, b_data.to_vec(), d_out)
        };

        // Track global rank (should be consistent across all layers).
        if let Some(prev) = rank_global {
            if prev != rank {
                eprintln!(
                    "[chat_metal] warning: rank mismatch across layers ({} vs {}); using first",
                    prev, rank
                );
            }
        } else {
            rank_global = Some(rank);
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

// ─── main logic ─────────────────────────────────────────────────────────────

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use std::io::Write;

    let args: Vec<String> = std::env::args().collect();

    // ── Arg parsing ──────────────────────────────────────────────────────────

    let emit_json = parse_flag(&args, "--json");

    let prompt_opt = parse_arg(&args, "--prompt");

    let max_tokens: usize = parse_arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(if emit_json { 512 } else { 512 });

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

    let is_q4_dir = !dir.join("model.safetensors").exists()
        && !dir.join("model.safetensors.index.json").exists()
        && std::fs::read_dir(dir)
            .ok()
            .and_then(|mut entries| {
                entries.find(|e| {
                    e.as_ref()
                        .ok()
                        .and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".q4")))
                        .unwrap_or(false)
                })
            })
            .is_some();

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
    let model_format; // "bf16" | "q4"

    if is_q4_dir {
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
        metal =
            MetalQwen35State::from_q4_dir(dir, &tokenizer_dir.join("tokenizer.json"), &cfg, 4096)
                .map_err(|e| format!("failed to initialize Metal from Q4 dir: {e}"))?;
        eprintln!(
            "[chat_metal] Q4 model loaded in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        model_format = "q4";
    } else {
        if !dir.join("model.safetensors").exists()
            && !dir.join("model.safetensors.index.json").exists()
        {
            return Err(format!(
                "no model found at {} (expected model.safetensors or .q4 files)",
                dir.display()
            )
            .into());
        }
        eprintln!("[chat_metal] Loading bf16 model from {}...", dir.display());
        let t0 = std::time::Instant::now();
        let model =
            Qwen35Model::from_safetensors(dir).map_err(|e| format!("failed to load model: {e}"))?;
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
        model_format = "bf16";
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
        stop_token_ids: vec![],
        enable_thinking: true,
        enable_mtp: None,
        grammar: None,
    };

    // ── JSON streaming mode (used by Lattice Studio app) ────────────────────

    if emit_json {
        let prompt = match prompt_opt {
            Some(p) => p,
            None => {
                return Err("--json mode requires --prompt".into());
            }
        };

        let mut stdout = std::io::stdout();
        let mut first_token_emitted = false;
        let mut ttft_ms: f64 = 0.0;
        let t1 = std::time::Instant::now();

        // generate_streaming: returns GenerateOutput (not Result), callback returns bool.
        let output = metal.generate_streaming(&prompt, &tokenizer, &gen_cfg, |delta, _token_id| {
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
            true // continue generation
        });

        let gen_ms = t1.elapsed().as_millis();
        let tok_s = if gen_ms > 0 {
            output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
        } else {
            0.0
        };

        // Final done event — format identical to generate_lora so app parser needs no change.
        writeln!(
            stdout,
            "@@lattice {{\"ev\":\"gen_token\",\"token\":\"\",\"done\":true,\"tok_s\":{tok_s:.1},\"ttft_ms\":{ttft_ms:.1}}}"
        )
        .ok();
        stdout.flush().ok();

        // Emit timing to stderr (not captured by app).
        let lora_tag = if has_lora { "+lora" } else { "" };
        eprintln!(
            "[chat_metal] GPU Metal {model_format}{lora_tag}: {} prompt + {} gen in {}ms = {tok_s:.1} tok/s",
            output.prompt_tokens, output.generated_tokens, gen_ms
        );

        return Ok(());
    }

    // ── Interactive REPL mode (no --json) ───────────────────────────────────

    use std::io::BufRead;

    let lora_tag = if has_lora { "+LoRA" } else { "" };
    eprintln!("\n=== GPU Metal {model_format}{lora_tag} — Qwen3.5 Chat ===");
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
        metal.reset_state();

        let t = std::time::Instant::now();

        // Use chat_completion_streaming for the REPL so output appears token-by-token.
        let mut response_text = String::new();
        let result = metal.chat_completion_streaming(&history, &tokenizer, &gen_cfg, |delta, _| {
            print!("{delta}");
            std::io::stdout().flush().ok();
            response_text.push_str(delta);
            true
        });
        println!();

        let elapsed = t.elapsed();
        let tps = if result.completion_tokens > 0 {
            result.completion_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "[{} prompt + {} gen in {:.1}ms = {:.1} tok/s | GPU Metal {model_format}]",
            result.prompt_tokens,
            result.completion_tokens,
            elapsed.as_secs_f64() * 1000.0,
            tps,
        );

        history.push(ChatMessage::assistant(response_text.trim()));
    }

    Ok(())
}
