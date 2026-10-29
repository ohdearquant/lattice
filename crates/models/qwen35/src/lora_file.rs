//! Load a PEFT or MLX LoRA adapter from a safetensors file.
//!
//! This is the file-format half of adapter support: it turns a path into the
//! `(layers, descriptor)` pair the Metal engine's `load_lora_adapter_with_descriptor`
//! takes. It lived inside `bin/chat_metal.rs` until the serving path needed to load an
//! adapter at runtime too, and a second copy of a format parser is a fork, not a reuse.
//!
//! The format itself is read by `lattice_lora::codec`, the same parser `lattice-tune`
//! uses: a bounded file read, the PEFT and MLX key forms of the text decoder, and, for
//! PEFT-form factors, the `adapter_config.json` beside the file.

/// Resolve the `(rank, alpha, scale)` a [`load_lora_safetensors`] call will
/// use, given the rank inferred from the adapter's tensor shapes and the
/// alpha the adapter declares (if any).
///
/// Prefers the declared alpha when available; falls back to `alpha = rank`
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

/// Load the adapter at `path` as Metal layer records and a descriptor.
///
/// Alpha comes from the PEFT `adapter_config.json` beside the file when the
/// factors are PEFT-form and that file exists, else from the safetensors
/// header, else it equals the rank. A malformed or unsupported config is an
/// error; only a missing one falls back. Factor tensors for a module name the
/// parser does not recognise are skipped and named in the load report.
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
    let (layers, descriptor, report) = load_lora_safetensors_with_report(path)?;
    let scale = lattice_fann::lora::effective_scale(descriptor.rank, report.alpha);
    eprintln!(
        "[chat_metal] LoRA: {} layer×module pairs, rank={}, alpha={} ({}), scale={:.2}",
        layers.len(),
        descriptor.rank,
        report.alpha,
        report.alpha_source,
        scale
    );
    if !report.skipped.is_empty() {
        eprintln!(
            "[chat_metal] LoRA: skipped {} tensors for unrecognised modules: {}",
            report.skipped.len(),
            report.skipped.join(", ")
        );
    }
    Ok((layers, descriptor))
}

/// Details of the factors and alpha selected while loading an adapter.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Debug, Clone)]
pub struct LoraLoadReport {
    /// Effective alpha used by the descriptor.
    pub alpha: f32,
    /// Source of the effective alpha.
    pub alpha_source: &'static str,
    /// Factor tensor names skipped because their modules are unrecognised.
    pub skipped: Vec<String>,
}

/// Load an adapter without logging and return its decode report.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn load_lora_safetensors_with_report(
    path: &std::path::Path,
) -> Result<
    (
        Vec<crate::forward::metal_qwen35::LoraLayerData>,
        lattice_fann::lora::LoraDescriptor,
        LoraLoadReport,
    ),
    Box<dyn std::error::Error>,
> {
    use crate::forward::metal_qwen35::LoraLayerData;
    use lattice_lora::codec;

    let bytes = codec::read_adapter_file_bounded(path, codec::MAX_ADAPTER_FILE_BYTES)?;
    let mut decoded = codec::decode_peft_safetensors(&bytes)?;
    if !decoded.has_mlx_factors {
        codec::apply_peft_sidecar_from_path(&mut decoded, path)?;
    }

    let mut layers: Vec<LoraLayerData> = decoded
        .layers
        .into_iter()
        .map(|((layer_idx, module), layer)| LoraLayerData {
            layer_idx,
            module,
            a: layer.a,
            b: layer.b,
            rank: layer.rank,
            d_in: layer.d_in,
            d_out: layer.d_out,
        })
        .collect();
    layers.sort_by(|x, y| (x.layer_idx, &x.module).cmp(&(y.layer_idx, &y.module)));

    let (rank, alpha, _) = resolve_lora_rank_alpha_scale(Some(decoded.rank), Some(decoded.alpha));

    // In-memory representation is always f32 (the codec converts F16/BF16
    // source tensors), matching the `lattice-tune` safetensors loader's own
    // convention for this field.
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank,
        alpha,
        target_modules: decoded.target_modules,
        dtype: "f32".into(),
    };

    let alpha_source = match decoded.alpha_source {
        codec::AlphaSource::Header => "safetensors header",
        codec::AlphaSource::PeftSidecar => "adapter_config.json",
        codec::AlphaSource::RankDefault => "rank default",
        _ => "unknown",
    };
    let report = LoraLoadReport {
        alpha,
        alpha_source,
        skipped: decoded.skipped,
    };
    Ok((layers, descriptor, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_lora::codec;
    use serde_json::json;
    use std::path::Path;

    fn adapter_bytes() -> Vec<u8> {
        let header = json!({
            "model.layers.0.self_attn.q_proj.lora_A": {"dtype":"F32", "shape":[1,1], "data_offsets":[0,4]},
            "model.layers.0.self_attn.q_proj.lora_B": {"dtype":"F32", "shape":[1,1], "data_offsets":[4,8]},
            "model.layers.0.self_attn.future_proj.lora_A": {"dtype":"F32", "shape":[1,1], "data_offsets":[8,12]},
            "model.layers.0.self_attn.future_proj.lora_B": {"dtype":"F32", "shape":[1,1], "data_offsets":[12,16]}
        });
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(&header);
        for value in [0.5f32, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        data
    }

    fn legacy_result(path: &Path) -> Result<codec::DecodedAdapter, lattice_lora::LoraError> {
        let bytes = codec::read_adapter_file_bounded(path, codec::MAX_ADAPTER_FILE_BYTES)?;
        let mut decoded = codec::decode_peft_safetensors(&bytes)?;
        if !decoded.has_mlx_factors
            && let Some(sidecar) = codec::read_peft_sidecar(path)?
        {
            codec::apply_peft_sidecar(&mut decoded, &sidecar)?;
        }
        Ok(decoded)
    }

    #[test]
    fn prepared_qwen_load_preserves_skips_alpha_and_factors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");
        std::fs::write(&path, adapter_bytes()).unwrap();
        std::fs::write(
            dir.path().join("adapter_config.json"),
            r#"{"lora_alpha":8,"r":1,"target_modules":".*q_proj","exclude_modules":".*v_proj"}"#,
        )
        .unwrap();
        let legacy = legacy_result(&path).unwrap();
        let (layers, descriptor, report) = load_lora_safetensors_with_report(&path).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(
            (layers[0].layer_idx, layers[0].module.as_str()),
            (0, "q_proj")
        );
        assert_eq!(layers[0].a, [0.5]);
        assert_eq!(layers[0].b, [2.0]);
        assert_eq!(descriptor.rank, legacy.rank);
        assert_eq!(descriptor.alpha.to_bits(), legacy.alpha.to_bits());
        assert_eq!(report.alpha.to_bits(), legacy.alpha.to_bits());
        assert_eq!(report.alpha_source, "adapter_config.json");
        assert_eq!(report.skipped, legacy.skipped);
        assert_eq!(report.skipped.len(), 2);
        assert!(
            report
                .skipped
                .iter()
                .all(|name| name.contains("future_proj"))
        );
    }

    #[test]
    fn prepared_qwen_load_preserves_missing_and_refusal_behavior() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");
        std::fs::write(&path, adapter_bytes()).unwrap();
        let (_, _, report) = load_lora_safetensors_with_report(&path).unwrap();
        assert_eq!(report.alpha, 1.0);
        assert_eq!(report.alpha_source, "rank default");
        assert_eq!(report.skipped.len(), 2);
        for sidecar in [
            r#"{"lora_alpha":8,"r":2,"target_modules":".*q_proj"}"#,
            r#"{"lora_alpha":8,"r":1,"target_modules":".*v_proj"}"#,
            r#"{"lora_alpha":8,"r":1,"target_modules":".*q_proj","exclude_modules":".*q_proj"}"#,
            r#"{"lora_alpha":8,"r":1,"target_modules":"q_proj)|(?:.*"}"#,
            "{malformed",
        ] {
            std::fs::write(dir.path().join("adapter_config.json"), sidecar).unwrap();
            let expected = legacy_result(&path).unwrap_err().to_string();
            let actual = match load_lora_safetensors_with_report(&path) {
                Ok(_) => panic!("sidecar unexpectedly accepted: {sidecar}"),
                Err(error) => error.to_string(),
            };
            assert_eq!(actual, expected, "sidecar {sidecar}");
        }
    }
}
