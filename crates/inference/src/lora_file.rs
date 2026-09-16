//! Load a PEFT or MLX LoRA adapter from a safetensors file.
//!
//! This is the file-format half of adapter support: it turns a path into the
//! `(layers, descriptor)` pair the Metal engine's `load_lora_adapter_with_descriptor`
//! takes. It lived inside `bin/chat_metal.rs` until the serving path needed to load an
//! adapter at runtime too, and a second copy of a format parser is a fork, not a reuse.
//!
//! No `lattice-tune` dependency: the parsing is done here against the two key layouts
//! the trainers emit, which is why the original carried the "inline" note.

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
fn is_supported_lora_target(block: &str, module: &str) -> bool {
    match block {
        "self_attn" => true,
        "mlp" => matches!(module, "gate_proj" | "up_proj" | "down_proj"),
        _ => false,
    }
}

/// Resolve the `(rank, alpha, scale)` a [`load_lora_safetensors`] call will
/// use, given the rank inferred from the adapter's tensor shapes and the
/// alpha parsed from `__metadata__` (if present).
///
/// Prefers `__metadata__.alpha` when available; falls back to `alpha = rank`
/// (scale = 1.0), matching the tune crate's own default for adapters
/// without embedded metadata. Scale itself is the shared
/// `lattice-fann` helper (issue #615) — a rank of 0 (malformed metadata)
/// resolves to scale = 0.0, matching `lattice_tune::lora::LoraConfig::scale`.
/// This bin previously fell back to scale = 1.0 for that case; adopting the
/// shared helper here resolves the drift to tune's value.
pub fn resolve_lora_rank_alpha_scale(
    rank_global: Option<usize>,
    metadata_alpha: Option<f32>,
) -> (usize, f32, f32) {
    let rank = rank_global.unwrap_or(1);
    let alpha = metadata_alpha.unwrap_or(rank as f32);
    let scale = lattice_fann::lora::effective_scale(rank, alpha);
    (rank, alpha, scale)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn load_lora_safetensors(
    path: &std::path::Path,
) -> Result<
    (
        Vec<crate::forward::metal_qwen35::LoraLayerData>,
        lattice_fann::lora::LoraDescriptor,
    ),
    Box<dyn std::error::Error>,
> {
    use crate::forward::metal_qwen35::LoraLayerData;
    use crate::weights::f32_weights::SafetensorsFile;

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
                && is_supported_lora_target(parts[1], parts[2])
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
                && is_supported_lora_target(parts[1], parts[2])
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

    let (rank, alpha, scale) = resolve_lora_rank_alpha_scale(rank_global, metadata_alpha);

    let mut target_modules: Vec<String> = layers.iter().map(|l| l.module.clone()).collect();
    target_modules.sort();
    target_modules.dedup();

    // In-memory representation is always f32 (`get_f32_tensor` above already
    // converted any F16/BF16 source tensors), matching the `lattice-tune`
    // safetensors loader's own convention for this field.
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank,
        alpha,
        target_modules,
        dtype: "f32".into(),
    };

    eprintln!(
        "[chat_metal] LoRA: {} layer×module pairs, rank={}, alpha={}, scale={:.2}",
        layers.len(),
        rank,
        alpha,
        scale
    );

    Ok((layers, descriptor))
}
