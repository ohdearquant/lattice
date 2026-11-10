//! Stream quantizer: converts sharded BF16 safetensors → Q4_0 `.q4` files.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin quantize_q4 -- \
//!   --model-dir ~/.lattice/models/qwen3.6-27b \
//!   --output-dir ~/.lattice/models/qwen3.6-27b-q4
//! ```
//!
//! # Memory budget
//!
//! Each iteration materializes one whole tensor as `f32` alongside its Q4
//! or F16 output, without staging a full `f64` tensor.

use lattice_inference::quant::quarot::QuarotTensorReader;
use lattice_weights::q4_weights::{Q4_BLOCK_BYTES, quantize_f32_to_q4, save_q4_file};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lattice_arch::qwen35_tensors::q4_tensor_stem;
use lattice_weights::f16_encode;

// ---------------------------------------------------------------------------
// Tensor classification: should_quantize
// ---------------------------------------------------------------------------

/// Returns `true` for large weight matrices that benefit from Q4_0 quantization.
///
/// Rule: quantize weight matrices for projections, MLP layers, embeddings, and lm_head.
/// Keep scalars, norms, biases, conv1d weights, and Mamba-specific parameters in f16.
fn should_quantize(name: &str) -> bool {
    // MoE routed-expert tensors are stored as one fused array per layer,
    // shape `[num_experts, out_features, in_features]`, WITHOUT a trailing
    // `.weight` suffix (e.g. `model.language_model.layers.0.mlp.experts
    // .gate_up_proj` / `.down_proj`, and the analogous `mtp.layers.N.mlp
    // .experts.*` tensors). They fail the `.weight`-suffix gate below and
    // were silently skipped, even though they hold the large majority of
    // parameters in a MoE checkpoint (~92% for a 256-expert model). Match
    // them explicitly before the suffix gate. The sibling `shared_expert.*`
    // and `shared_expert_gate` tensors already carry a `.weight` suffix and
    // are already covered by the rules below.
    if name.ends_with(".experts.gate_up_proj") || name.ends_with(".experts.down_proj") {
        return true;
    }

    // Must be a weight tensor.
    if !name.ends_with(".weight") && !name.ends_with("lm_head.weight") {
        return false;
    }

    // Always quantize these large matrices.
    if name.ends_with("_proj.weight")
        || name.ends_with("_proj_a.weight")
        || name.ends_with("_proj_b.weight")
        || name.ends_with("_proj_qkv.weight")
        || name.ends_with("_proj_z.weight")
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with("down_proj.weight")
        || name.ends_with("lm_head.weight")
        || name.ends_with("embed_tokens.weight")
    {
        return true;
    }

    // Keep small / special tensors in f16 (norms, conv1d, biases).
    // These checks shadow the weight-check above for norm weights, which are small.
    if name.contains("norm.weight")
        || name.contains("norm_")
        || name.ends_with(".bias")
        || name.ends_with("A_log")
        || name.ends_with("dt_bias")
        || name.ends_with("conv1d.weight")
    {
        return false;
    }

    // Default: quantize unknown weight matrices.
    true
}

// ---------------------------------------------------------------------------
// Output index
// ---------------------------------------------------------------------------

/// Index entry recorded in `quantize_index.json`.
#[derive(serde::Serialize)]
struct IndexEntry {
    /// Source tensor name.
    name: String,
    /// Output file stem (relative to output directory).
    file: String,
    /// Whether the tensor was quantized (true) or saved as f16 (false).
    quantized: bool,
    /// Original shape.
    shape: Vec<usize>,
    /// Number of original elements.
    numel: usize,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn print_usage_and_exit() -> ! {
    eprintln!("Usage: quantize_q4 --model-dir <DIR> --output-dir <DIR> [--dry-run]");
    eprintln!();
    eprintln!("  --model-dir   directory containing model.safetensors[.index.json]");
    eprintln!("  --output-dir  directory to write .q4 and index files");
    eprintln!("  --dry-run     read tensors but skip writing output");
    std::process::exit(1);
}

fn main() {
    if let Err(e) = run() {
        eprintln!("quantize_q4 failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut dry_run = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("--model-dir requires an argument");
                    print_usage_and_exit();
                })));
            }
            "--output-dir" => {
                i += 1;
                output_dir = Some(PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("--output-dir requires an argument");
                    print_usage_and_exit();
                })));
            }
            "--dry-run" => dry_run = true,
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage_and_exit();
            }
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(|| {
        eprintln!("--model-dir is required");
        print_usage_and_exit();
    });
    let output_dir = output_dir.unwrap_or_else(|| {
        eprintln!("--output-dir is required");
        print_usage_and_exit();
    });

    quantize_model(&model_dir, &output_dir, dry_run)
}

fn quantize_model(
    model_dir: &Path,
    output_dir: &Path,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !dry_run {
        fs::create_dir_all(output_dir)?;
        let source_config = model_dir.join("config.json");
        let output_config = output_dir.join("config.json");
        fs::copy(&source_config, &output_config).map_err(|e| {
            format!(
                "failed to copy {} to {}: {e}",
                source_config.display(),
                output_config.display()
            )
        })?;
    }

    let reader = QuarotTensorReader::open(model_dir)?;
    let mut tensor_names = reader.tensor_names();
    tensor_names.sort();
    let n_tensors = tensor_names.len();

    eprintln!("=== quantize_q4: SafeTensors → Q4_0 ===");
    eprintln!("Model dir:  {}", model_dir.display());
    eprintln!("Output dir: {}", output_dir.display());
    eprintln!("Tensors:    {n_tensors}");
    if dry_run {
        eprintln!("Mode:       DRY RUN (no files written)");
    }
    eprintln!();

    let global_start = Instant::now();
    let mut index_entries: Vec<IndexEntry> = Vec::new();
    let mut total_tensors = 0usize;
    let mut total_quantized = 0usize;
    let mut total_kept_f16 = 0usize;
    let mut total_bytes_in = 0u64;
    let mut total_bytes_out = 0u64;

    for (tensor_idx, tensor_name) in tensor_names.iter().enumerate() {
        let tensor_start = Instant::now();
        let bytes_in = reader.tensor_byte_len(tensor_name)?;
        let source_dtype = reader.source_dtype(tensor_name)?;
        let (data_f32, shape) = reader.read_tensor_f32(tensor_name)?;

        let expected_numel = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| format!("tensor {tensor_name}: shape product overflow for {shape:?}"))?;
        if expected_numel != data_f32.len() {
            return Err(format!(
                "tensor {tensor_name}: shape {shape:?} has {expected_numel} elements, \
                 reader returned {}",
                data_f32.len()
            )
            .into());
        }

        let numel = data_f32.len();
        total_bytes_in += bytes_in;

        let stem = q4_tensor_stem(tensor_name);

        if should_quantize(tensor_name) {
            let q4 = quantize_f32_to_q4(&data_f32, &shape).map_err(|error| {
                format!(
                    "{}: tensor {tensor_name} failed Q4 encoding: {error}",
                    model_dir.display(),
                )
            })?;
            let bytes_out = (q4.blocks.len() * Q4_BLOCK_BYTES) as u64;
            total_bytes_out += bytes_out;

            let out_filename = format!("{stem}.q4");
            let out_path = output_dir.join(&out_filename);

            if !dry_run {
                save_q4_file(&out_path, &q4)
                    .map_err(|e| format!("failed to write {}: {e}", out_path.display()))?;
            }

            let elapsed = tensor_start.elapsed();
            eprintln!(
                "  [{}/{n_tensors}] Q4_0  {tensor_name}  shape={shape:?}  \
                 {:.1}MB→{:.1}MB  {:.2}s",
                tensor_idx + 1,
                bytes_in as f64 / 1_048_576.0,
                bytes_out as f64 / 1_048_576.0,
                elapsed.as_secs_f64()
            );

            index_entries.push(IndexEntry {
                name: tensor_name.clone(),
                file: out_filename,
                quantized: true,
                shape: shape.clone(),
                numel,
            });
            total_quantized += 1;
        } else {
            // Kept tensor: reader already decoded to numeric values, so the
            // common path is decoded-value → f16 for every source dtype.
            let mut f16_data = Vec::with_capacity(data_f32.len() * 2);
            for (index, &value) in data_f32.iter().enumerate() {
                let bits = f16_encode::f32_to_finite_f16_bits(value).map_err(|bits| {
                    format!(
                        "{}: tensor {tensor_name} cannot encode value {value} at element \
                         index {index} as finite F16 (encoded bits {bits:#06x})",
                        model_dir.display(),
                    )
                })?;
                f16_data.extend_from_slice(&bits.to_le_bytes());
            }

            let bytes_out = f16_data.len() as u64;
            total_bytes_out += bytes_out;

            let out_filename = format!("{stem}.f16");
            let out_path = output_dir.join(&out_filename);

            if !dry_run {
                let mut f = fs::File::create(&out_path)
                    .map_err(|e| format!("failed to create {}: {e}", out_path.display()))?;
                // Minimal header: magic "KHF1" + version u32 + ndim u32 + shape[i] u64 + numel u64 + data
                f.write_all(b"KHF1")?;
                f.write_all(&1u32.to_le_bytes())?;
                f.write_all(&(shape.len() as u32).to_le_bytes())?;
                for &dim in &shape {
                    f.write_all(&(dim as u64).to_le_bytes())?;
                }
                f.write_all(&(numel as u64).to_le_bytes())?;
                f.write_all(&f16_data)?;
            }

            let elapsed = tensor_start.elapsed();
            eprintln!(
                "  [{}/{n_tensors}] F16   {tensor_name}  shape={shape:?}  \
                 {:.1}MB  dtype={}  {:.3}s",
                tensor_idx + 1,
                bytes_in as f64 / 1_048_576.0,
                source_dtype.name(),
                elapsed.as_secs_f64()
            );

            index_entries.push(IndexEntry {
                name: tensor_name.clone(),
                file: out_filename,
                quantized: false,
                shape: shape.clone(),
                numel,
            });
            total_kept_f16 += 1;
        }

        total_tensors += 1;
    }

    // Write the quantization index.
    if !dry_run {
        let index_path = output_dir.join("quantize_index.json");
        let index_json = serde_json::to_string_pretty(&index_entries)
            .map_err(|e| format!("failed to serialize index: {e}"))?;
        fs::write(&index_path, index_json)
            .map_err(|e| format!("failed to write {}: {e}", index_path.display()))?;
        eprintln!("Index written: {}", index_path.display());
    }

    let total_elapsed = global_start.elapsed();
    let compression = if total_bytes_in > 0 {
        total_bytes_out as f64 / total_bytes_in as f64
    } else {
        1.0
    };

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("Tensors processed: {total_tensors}");
    eprintln!("  Quantized (Q4_0): {total_quantized}");
    eprintln!("  Kept (F16):       {total_kept_f16}");
    eprintln!(
        "Input size:   {:.2} GB",
        total_bytes_in as f64 / 1_073_741_824.0
    );
    eprintln!(
        "Output size:  {:.2} GB",
        total_bytes_out as f64 / 1_073_741_824.0
    );
    eprintln!(
        "Ratio:        {:.2}x  ({:.1}%)",
        1.0 / compression,
        compression * 100.0
    );
    eprintln!("Total time:   {:.1}s", total_elapsed.as_secs_f64());

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{quantize_model, should_quantize};

    fn write_f32_tensor_model(
        model_dir: &std::path::Path,
        tensors: &[(&str, &[f32])],
    ) -> std::path::PathBuf {
        std::fs::create_dir_all(model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (tensor_name, values) in tensors {
            let start = payload.len();
            for value in *values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let end = payload.len();
            header.insert(
                (*tensor_name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [values.len()],
                    "data_offsets": [start, end],
                }),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(header.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(&header);
        safetensors.extend_from_slice(&payload);
        let source_path = model_dir.join("model.safetensors");
        std::fs::write(&source_path, safetensors).unwrap();
        source_path
    }

    #[test]
    fn later_invalid_source_tensor_is_not_published_after_valid_tensor() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        let output_dir = tmp.path().join("output");
        let valid_name = "a_valid.norm.weight";
        let invalid_name = "z_invalid.norm.weight";
        let source_path = write_f32_tensor_model(
            &model_dir,
            &[(valid_name, &[1.0]), (invalid_name, &[1.0, f32::NAN])],
        );

        let err = quantize_model(&model_dir, &output_dir, false).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(&source_path.display().to_string()),
            "{message}"
        );
        assert!(message.contains(invalid_name), "{message}");
        assert!(message.contains("non-finite value"), "{message}");

        assert!(
            output_dir.join("a_valid_norm_weight.f16").exists(),
            "an earlier independently valid tensor may remain published"
        );
        assert!(
            !output_dir.join("z_invalid_norm_weight.f16").exists(),
            "the tensor that failed validation must not be published"
        );
        assert!(
            !output_dir.join("quantize_index.json").exists(),
            "failed conversion must not publish an index"
        );
    }

    #[test]
    fn finite_f32_that_overflows_f16_is_not_published() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        let output_dir = tmp.path().join("output");
        let tensor_name = "model.language_model.norm.weight";
        write_f32_tensor_model(&model_dir, &[(tensor_name, &[100_000.0])]);

        let err = quantize_model(&model_dir, &output_dir, false).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(&model_dir.display().to_string()),
            "{message}"
        );
        assert!(message.contains(tensor_name), "{message}");
        assert!(message.contains("cannot encode value"), "{message}");
        assert!(message.contains("as finite F16"), "{message}");
        assert!(
            std::fs::read_dir(&output_dir)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .all(|path| path.extension().is_none_or(|extension| extension != "f16")),
            "overflowed F16 payload must not be published"
        );
    }

    #[test]
    fn finite_extreme_quantized_tensor_is_not_published() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        let output_dir = tmp.path().join("output");
        let tensor_name = "model.layers.0.mlp.gate_proj.weight";
        let values = [f32::MAX; 32];
        write_f32_tensor_model(&model_dir, &[(tensor_name, &values)]);

        let err = quantize_model(&model_dir, &output_dir, false).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(&model_dir.display().to_string()),
            "{message}"
        );
        assert!(message.contains(tensor_name), "{message}");
        assert!(
            !output_dir
                .join("model_layers_0_mlp_gate_proj_weight.q4")
                .exists()
        );
        assert!(!output_dir.join("quantize_index.json").exists());
    }

    // -----------------------------------------------------------------------
    // MoE routed-expert tensors (issue #874 regression coverage).
    //
    // Mutation-sensitive: these tensor names have no `.weight` suffix, so the
    // pre-fix gate (`if !name.ends_with(".weight") ... return false`) rejects
    // them before reaching any of the quantize-candidate checks. Reverting
    // the `.experts.gate_up_proj` / `.experts.down_proj` special-case added
    // in this fix makes every assertion below fail.
    // -----------------------------------------------------------------------

    #[test]
    fn should_quantize_accepts_routed_expert_tensors_main_layers() {
        assert!(should_quantize(
            "model.language_model.layers.0.mlp.experts.gate_up_proj"
        ));
        assert!(should_quantize(
            "model.language_model.layers.0.mlp.experts.down_proj"
        ));
        assert!(should_quantize(
            "model.language_model.layers.39.mlp.experts.gate_up_proj"
        ));
        assert!(should_quantize(
            "model.language_model.layers.39.mlp.experts.down_proj"
        ));
    }

    #[test]
    fn should_quantize_accepts_routed_expert_tensors_mtp_layer() {
        // The speculative-decode MTP head has its own MoE layer under a
        // different name prefix; the suffix match must not be anchored to
        // the `model.language_model.` prefix.
        assert!(should_quantize("mtp.layers.0.mlp.experts.gate_up_proj"));
        assert!(should_quantize("mtp.layers.0.mlp.experts.down_proj"));
    }

    #[test]
    fn should_quantize_rejects_names_that_merely_contain_experts() {
        // Precision check: the match is a name-suffix match, not a substring
        // match, so a hypothetical `.experts.gate_up_proj_extra` (or any
        // other name that merely contains "experts") does not get swept in
        // by accident.
        assert!(!should_quantize(
            "model.language_model.layers.0.mlp.experts.gate_up_proj_extra"
        ));
    }

    // -----------------------------------------------------------------------
    // Dense / already-suffixed tensors — must be unaffected by the fix.
    // -----------------------------------------------------------------------

    #[test]
    fn should_quantize_accepts_dense_projection_weights() {
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert!(should_quantize(name), "expected quantize=true for {name}");
        }
    }

    #[test]
    fn should_quantize_accepts_shared_expert_weights() {
        // The shared (always-on) expert and its gate already carry a
        // `.weight` suffix and were already quantized correctly pre-fix;
        // this pins that they remain quantized post-fix.
        for name in [
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        ] {
            assert!(should_quantize(name), "expected quantize=true for {name}");
        }
    }

    #[test]
    fn should_quantize_accepts_router_gate_weight() {
        // The per-layer MoE router (`mlp.gate.weight`, shape [num_experts,
        // hidden]) is small relative to the routed experts but already
        // falls through to the "quantize unknown weight matrices" default;
        // this pins that unrelated behavior stays unchanged.
        assert!(should_quantize(
            "model.language_model.layers.0.mlp.gate.weight"
        ));
    }

    // -----------------------------------------------------------------------
    // Norms / biases / Mamba-specific scalars — must stay excluded.
    // -----------------------------------------------------------------------

    #[test]
    fn should_quantize_rejects_norms_biases_and_mamba_scalars() {
        for name in [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            "model.language_model.layers.0.self_attn.o_proj.bias",
            "model.language_model.layers.0.mamba.A_log",
            "model.language_model.layers.0.mamba.dt_bias",
            "model.language_model.layers.0.mamba.conv1d.weight",
        ] {
            assert!(!should_quantize(name), "expected quantize=false for {name}");
        }
    }

    #[test]
    fn should_quantize_rejects_non_weight_non_expert_tensors() {
        // Anything that isn't a `.weight` tensor and isn't a recognized
        // suffix-less expert tensor must be rejected outright.
        assert!(!should_quantize(
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq"
        ));
    }

    fn write_raw_half_tensor_model(
        model_dir: &std::path::Path,
        dtype: &str,
        tensors: &[(&str, &[u16])],
    ) -> std::path::PathBuf {
        assert!(matches!(dtype, "F16" | "BF16"));
        std::fs::create_dir_all(model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (tensor_name, values) in tensors {
            let start = payload.len();
            for value in *values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let end = payload.len();
            header.insert(
                (*tensor_name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": [values.len()],
                    "data_offsets": [start, end],
                }),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(header.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(&header);
        safetensors.extend_from_slice(&payload);
        let source_path = model_dir.join("model.safetensors");
        std::fs::write(&source_path, safetensors).unwrap();
        source_path
    }

    fn write_f16_tensor_model(
        model_dir: &std::path::Path,
        tensors: &[(&str, &[u16])],
    ) -> std::path::PathBuf {
        write_raw_half_tensor_model(model_dir, "F16", tensors)
    }

    fn write_bf16_tensor_model(
        model_dir: &std::path::Path,
        tensors: &[(&str, &[u16])],
    ) -> std::path::PathBuf {
        write_raw_half_tensor_model(model_dir, "BF16", tensors)
    }

    #[test]
    #[ignore = "requires a fresh absolute LATTICE_QUANTIZE_PEAK_FIXTURE_ROOT"]
    fn export_quantize_peak_fixtures() {
        let root = std::path::PathBuf::from(
            std::env::var_os("LATTICE_QUANTIZE_PEAK_FIXTURE_ROOT")
                .expect("LATTICE_QUANTIZE_PEAK_FIXTURE_ROOT must be explicitly set"),
        );
        assert!(root.is_absolute(), "fixture root must be absolute");
        std::fs::create_dir(&root)
            .expect("fixture root must not exist and its parent must already exist");

        let tensor_names = [
            (
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.input_layernorm.weight",
            ),
            (
                "model.layers.1.mlp.gate_proj.weight",
                "model.layers.1.input_layernorm.weight",
            ),
            (
                "model.layers.2.mlp.gate_proj.weight",
                "model.layers.2.input_layernorm.weight",
            ),
        ];
        let mut f32_projection_bits = [0u32; 33];
        f32_projection_bits[1] = 0x4170_0000;
        f32_projection_bits[2] = 0x3eff_ffff;
        f32_projection_bits[32] = 0x40e0_0000;
        let f32_projection = f32_projection_bits.map(f32::from_bits);
        let f32_norm = [
            0x0000_0000,
            0x8000_0000,
            0x3380_0000,
            0xb380_0000,
            0x3f00_0000,
            0xbf00_0000,
            0x3f80_0000,
            0xbf80_0000,
        ]
        .map(f32::from_bits);
        write_f32_tensor_model(
            &root.join("single-f32"),
            &[
                (tensor_names[0].0, &f32_projection),
                (tensor_names[0].1, &f32_norm),
            ],
        );

        let mut f16_projection = [
            0xc000u16, 0xbc00, 0x8000, 0x0000, 0x3800, 0x3c00, 0x4000, 0x4200,
        ]
        .repeat(4);
        f16_projection.push(0x4700);
        let f16_norm = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x3800, 0xb800, 0x3c00, 0xbc00,
        ];
        write_f16_tensor_model(
            &root.join("single-f16"),
            &[
                (tensor_names[1].0, &f16_projection),
                (tensor_names[1].1, &f16_norm),
            ],
        );

        let mut bf16_projection = [
            0xc000u16, 0xbf80, 0x8000, 0x0000, 0x3f00, 0x3f80, 0x4000, 0x4040,
        ]
        .repeat(4);
        bf16_projection.push(0x40e0);
        let bf16_norm = [
            0x0000, 0x8000, 0x3380, 0xb380, 0x3f00, 0xbf00, 0x3f80, 0xbf80,
        ];
        write_bf16_tensor_model(
            &root.join("single-bf16"),
            &[
                (tensor_names[2].0, &bf16_projection),
                (tensor_names[2].1, &bf16_norm),
            ],
        );

        let mixed_dir = root.join("sharded-mixed");
        std::fs::create_dir(&mixed_dir).unwrap();
        std::fs::write(mixed_dir.join("config.json"), b"{}").unwrap();
        let single_cases = [
            ("single-f32", "F32", "model-f32.safetensors"),
            ("single-f16", "F16", "model-f16.safetensors"),
            ("single-bf16", "BF16", "model-bf16.safetensors"),
        ];
        let mut cases = Vec::new();
        let mut weight_map = serde_json::Map::new();
        let mut mixed_tensors = Vec::new();
        let mut mixed_outputs = vec!["config.json".to_string(), "quantize_index.json".to_string()];
        for ((case_name, dtype, shard_name), (projection_name, norm_name)) in
            single_cases.into_iter().zip(tensor_names)
        {
            let source_path = root.join(case_name).join("model.safetensors");
            let shard_path = mixed_dir.join(shard_name);
            std::fs::copy(&source_path, &shard_path).unwrap();
            assert_eq!(
                std::fs::read(&source_path).unwrap(),
                std::fs::read(&shard_path).unwrap(),
                "mixed shard must preserve the single-case bytes"
            );
            let projection_output = format!("{}.q4", projection_name.replace('.', "_"));
            let norm_output = format!("{}.f16", norm_name.replace('.', "_"));
            mixed_outputs.push(projection_output.clone());
            mixed_outputs.push(norm_output.clone());
            let tensors = vec![
                serde_json::json!({"name": projection_name, "dtype": dtype, "shape": [33]}),
                serde_json::json!({"name": norm_name, "dtype": dtype, "shape": [8]}),
            ];
            mixed_tensors.extend(tensors.iter().cloned());
            cases.push(serde_json::json!({
                "name": case_name,
                "source_files": ["config.json", "model.safetensors"],
                "outputs": ["config.json", "quantize_index.json", projection_output, norm_output],
                "tensors": tensors,
            }));
            for name in [projection_name, norm_name] {
                weight_map.insert(name.to_string(), serde_json::json!(shard_name));
            }
        }
        std::fs::write(
            mixed_dir.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "metadata": {},
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
        cases.push(serde_json::json!({
            "name": "sharded-mixed",
            "source_files": [
                "config.json",
                "model.safetensors.index.json",
                "model-f32.safetensors",
                "model-f16.safetensors",
                "model-bf16.safetensors",
            ],
            "outputs": mixed_outputs,
            "tensors": mixed_tensors,
        }));
        assert_eq!(cases.len(), 4);
        let mut source_count = 0;
        let mut output_count = 0;
        for case in &cases {
            let case_name = case["name"].as_str().unwrap();
            let mut expected_sources: Vec<_> = case["source_files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| name.as_str().unwrap().to_string())
                .collect();
            let mut actual_sources: Vec<_> = std::fs::read_dir(root.join(case_name))
                .unwrap()
                .map(|entry| {
                    let entry = entry.unwrap();
                    assert!(entry.file_type().unwrap().is_file());
                    entry.file_name().into_string().unwrap()
                })
                .collect();
            expected_sources.sort();
            actual_sources.sort();
            assert_eq!(actual_sources, expected_sources, "case={case_name}");
            source_count += actual_sources.len();
            output_count += case["outputs"].as_array().unwrap().len();
        }
        assert_eq!(source_count, 11);
        assert_eq!(output_count, 20);
        std::fs::write(
            root.join("FIXTURES.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": "lattice-quantize-peak-fixtures-v1",
                "cases": cases,
            }))
            .unwrap(),
        )
        .unwrap();
        eprintln!(
            "R76_FIXTURES_EXPORTED root={} cases=4 model_source_files=11 expected_output_files=20",
            root.display()
        );
    }
}
