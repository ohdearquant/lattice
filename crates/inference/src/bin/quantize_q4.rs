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
//! At any point only one tensor's decoded `f64` values are live in RAM
//! alongside its `f32` downcast and Q4 output.

use lattice_inference::quant::quarot::QuarotTensorReader;
use lattice_inference::weights::q4_weights::{Q4_BLOCK_BYTES, quantize_f32_to_q4, save_q4_file};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

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
// F32 → F16 helper (for non-quantized tensors written as f16)
// ---------------------------------------------------------------------------

/// Convert f32 to IEEE-754 f16 bit pattern with round-to-nearest-even.
#[inline]
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        if frac == 0 {
            return sign | 0x7c00;
        }
        let mut payload = ((frac >> 13) as u16) & 0x03ff;
        if payload == 0 {
            payload = 1;
        }
        return sign | 0x7c00 | payload | 0x0200;
    }
    if exp == 0 {
        return sign;
    }

    let exp32 = exp - 127;
    if exp32 > 15 {
        return sign | 0x7c00;
    }

    if exp32 >= -14 {
        let frac16_raw = (frac >> 13) as u16;
        let round_bit = ((frac >> 12) & 1) as u16;
        let sticky = (frac & 0x0fff) != 0;
        let frac16 = frac16_raw
            + if round_bit == 1 && (sticky || (frac16_raw & 1) == 1) {
                1
            } else {
                0
            };
        let mut exp16 = (exp32 + 15) as u16;
        let mut frac16_final = frac16 & 0x03ff;
        if frac16 == 0x0400 {
            frac16_final = 0;
            exp16 += 1;
            if exp16 >= 0x1f {
                return sign | 0x7c00;
            }
        }
        return sign | (exp16 << 10) | frac16_final;
    }

    // exp32 < -14: the f32 value is normal-range but smaller than the
    // smallest f16 normal. F16 still represents 2^-24..2^-14 as subnormals,
    // and a source-F16 subnormal reaches this point after widening through
    // f64 and narrowing back to f32 (both exact), so flushing to zero here
    // would silently destroy a value f16 can represent exactly. Encode it
    // as an f16 subnormal instead, using the same round-to-nearest-even
    // shape as the normal-range path above.
    let sig = 0x0080_0000u32 | frac;
    let shift = (-exp32 - 1) as u32;

    // Beyond this shift the discarded bits can never reach halfway to the
    // smallest subnormal (2^-24), so the correctly rounded result is zero.
    if shift >= 25 {
        return sign;
    }

    let frac16_raw = (sig >> shift) as u16;
    let round_bit = ((sig >> (shift - 1)) & 1) as u16;
    let sticky = (sig & ((1u32 << (shift - 1)) - 1)) != 0;
    let frac16 = frac16_raw
        + if round_bit == 1 && (sticky || (frac16_raw & 1) == 1) {
            1
        } else {
            0
        };

    // frac16 == 0x0400 means rounding overflowed the largest subnormal into
    // the smallest normal f16 (exponent field 0x01, fraction 0) — that bit
    // pattern is exactly `sign | 0x0400`, so no separate branch is needed.
    sign | frac16
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

    if !dry_run {
        fs::create_dir_all(&output_dir)?;
    }

    let reader = QuarotTensorReader::open(&model_dir)?;
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
        let (data_f64, shape) = reader.read_tensor_f64(tensor_name)?;

        let expected_numel = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| format!("tensor {tensor_name}: shape product overflow for {shape:?}"))?;
        if expected_numel != data_f64.len() {
            return Err(format!(
                "tensor {tensor_name}: shape {shape:?} has {expected_numel} elements, \
                 reader returned {}",
                data_f64.len()
            )
            .into());
        }

        // Reader decodes to f64; the Q4 quantizer works in f32 (ADR-044 step 3c).
        let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();
        let numel = data_f32.len();
        total_bytes_in += bytes_in;

        let sanitized: String = tensor_name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();

        if should_quantize(tensor_name) {
            let q4 = quantize_f32_to_q4(&data_f32, &shape)?;
            let bytes_out = (q4.blocks.len() * Q4_BLOCK_BYTES) as u64;
            total_bytes_out += bytes_out;

            let out_filename = format!("{sanitized}.q4");
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
            let f16_data: Vec<u8> = data_f32
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();

            let bytes_out = f16_data.len() as u64;
            total_bytes_out += bytes_out;

            let out_filename = format!("{sanitized}.f16");
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
    use super::should_quantize;

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
}
