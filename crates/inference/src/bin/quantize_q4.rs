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
//! At any point only one BF16 tensor is live in RAM alongside its Q4 output.
//! Peak ≈ (tensor_elements × 6 bytes) — 178MB BF16 + 89MB Q4 for the largest shard tensor.

use lattice_inference::weights::f32_weights::parse_index;
use lattice_inference::weights::q4_weights::{quantize_bf16_to_q4, save_q4_file};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Minimal safetensors header parser (BF16 / F16 / F32 aware, no feature gate)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    fn bytes_per_elem(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        }
    }
}

#[derive(Debug)]
struct TensorHeader {
    dtype: DType,
    shape: Vec<usize>,
    /// Byte offsets *relative to the start of the data section* (after the 8-byte length prefix + header).
    start: usize,
    end: usize,
}

/// Open a safetensors shard file and return its memory map + parsed tensor headers.
fn open_shard(path: &Path) -> io::Result<(Mmap, HashMap<String, TensorHeader>)> {
    let file = fs::File::open(path)?;
    // SAFETY: The file is opened read-only. The Mmap owns the mapping; the File
    // can be dropped immediately after — the OS keeps the fd alive through the map.
    let mmap = unsafe { Mmap::map(&file)? };

    if mmap.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "safetensors file too small",
        ));
    }

    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_offset = 8 + header_len;
    if data_offset > mmap.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "header extends past end of file: header_end={data_offset}, file_len={}",
                mmap.len()
            ),
        ));
    }

    let header_str = std::str::from_utf8(&mmap[8..data_offset])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let root: Value = serde_json::from_str(header_str)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let obj = root.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "safetensors header is not an object",
        )
    })?;

    let mut tensors = HashMap::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = entry["dtype"]
            .as_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing dtype"))?;
        let dtype = match dtype_str {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            other => {
                // Warn but skip unsupported dtypes (e.g. I32, I64 used by tokenizers)
                eprintln!("  [skip] tensor {name}: unsupported dtype {other}");
                continue;
            }
        };

        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing shape"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-u64 shape dim"))
                    .map(|x| x as usize)
            })
            .collect::<io::Result<Vec<_>>>()?;

        let offsets = entry["data_offsets"]
            .as_array()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing data_offsets"))?;
        if offsets.len() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "data_offsets must have length 2",
            ));
        }
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;

        // Sanity check byte length vs shape × dtype.
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.bytes_per_elem();
        if end - start != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name}: byte length mismatch — shape {:?} × {} bytes = {} expected, got {}",
                    shape,
                    dtype.bytes_per_elem(),
                    expected_bytes,
                    end - start
                ),
            ));
        }

        tensors.insert(
            name.clone(),
            TensorHeader {
                dtype,
                shape,
                start,
                end,
            },
        );
    }

    // Re-open to bake in the data_offset — store it in a wrapper that knows it.
    // We return the raw mmap; callers use `data_offset` when indexing.
    // Pack data_offset into the mmap key 0..8 via a simple trick: we wrap mmap
    // in a struct below so we do NOT re-open.  The returned map uses data_offset
    // by slicing: mmap[data_offset + start .. data_offset + end].
    // We stash data_offset as an extra entry with key "" (impossible tensor name).
    let fake_start = data_offset; // callers read this back via SENTINEL_KEY
    tensors.insert(
        "\x00data_offset".to_string(),
        TensorHeader {
            dtype: DType::F32,
            shape: vec![],
            start: fake_start,
            end: fake_start,
        },
    );

    Ok((mmap, tensors))
}

/// Get the raw bytes for a tensor from a parsed shard.
///
/// `tensors` is the map from `open_shard`.
fn tensor_bytes<'m>(
    mmap: &'m Mmap,
    tensors: &HashMap<String, TensorHeader>,
    name: &str,
) -> &'m [u8] {
    let data_offset = tensors["\x00data_offset"].start;
    let h = &tensors[name];
    &mmap[data_offset + h.start..data_offset + h.end]
}

// ---------------------------------------------------------------------------
// Tensor classification: should_quantize
// ---------------------------------------------------------------------------

/// Returns `true` for large weight matrices that benefit from Q4_0 quantization.
///
/// Rule: quantize weight matrices for projections, MLP layers, embeddings, and lm_head.
/// Keep scalars, norms, biases, conv1d weights, and Mamba-specific parameters in f16.
fn should_quantize(name: &str) -> bool {
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

    sign
}

/// Convert BF16 bit pattern to f32.
#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
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
    eprintln!("  --model-dir   directory containing model.safetensors.index.json");
    eprintln!("  --output-dir  directory to write .q4 and index files");
    eprintln!("  --dry-run     parse shards but skip writing output");
    std::process::exit(1);
}

fn main() {
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
        fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
            panic!(
                "failed to create output directory {}: {e}",
                output_dir.display()
            );
        });
    }

    // Parse the shard index.
    let shard_index = parse_index(&model_dir).unwrap_or_else(|e| {
        panic!(
            "failed to parse model.safetensors.index.json in {}: {e}",
            model_dir.display()
        );
    });

    // Collect unique shard filenames in sorted order (model-00001-of-00015, ...).
    let mut shard_files: Vec<String> = {
        let mut seen: std::collections::BTreeSet<String> = Default::default();
        for v in shard_index.weight_map.values() {
            seen.insert(v.clone());
        }
        seen.into_iter().collect()
    };
    shard_files.sort();
    let n_shards = shard_files.len();

    // Build reverse map: shard_filename → [tensor_names in that shard].
    let mut shard_to_tensors: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, shard_name) in &shard_index.weight_map {
        shard_to_tensors
            .entry(shard_name.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    eprintln!("=== quantize_q4: BF16 → Q4_0 ===");
    eprintln!("Model dir:  {}", model_dir.display());
    eprintln!("Output dir: {}", output_dir.display());
    eprintln!("Shards:     {n_shards}");
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

    for (shard_idx, shard_filename) in shard_files.iter().enumerate() {
        let shard_path = model_dir.join(shard_filename);
        eprintln!(
            "[shard {}/{n_shards}] Opening {}",
            shard_idx + 1,
            shard_path.display()
        );
        let shard_start = Instant::now();

        let (mmap, headers) = open_shard(&shard_path).unwrap_or_else(|e| {
            panic!("failed to open shard {}: {e}", shard_path.display());
        });

        // Sort tensors in this shard by their offset for sequential access.
        let mut shard_tensors = shard_to_tensors
            .get(shard_filename)
            .cloned()
            .unwrap_or_default();
        shard_tensors.sort_by_key(|name| headers.get(name).map(|h| h.start).unwrap_or(usize::MAX));

        let n_in_shard = shard_tensors.len();
        eprintln!("  Tensors in shard: {n_in_shard}");

        for (tensor_idx, tensor_name) in shard_tensors.iter().enumerate() {
            let Some(h) = headers.get(tensor_name.as_str()) else {
                eprintln!(
                    "  [warn] tensor {tensor_name} listed in index but not found in shard header — skipping"
                );
                continue;
            };

            let shape = &h.shape;
            let numel: usize = shape.iter().product();
            let bytes_in = (h.end - h.start) as u64;
            total_bytes_in += bytes_in;

            let tensor_start = Instant::now();
            let raw_bytes = tensor_bytes(&mmap, &headers, tensor_name);

            if should_quantize(tensor_name) {
                // BF16 → f32 → Q4_0
                let bf16_vals: Vec<u16> = raw_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();

                let q4 = quantize_bf16_to_q4(&bf16_vals, shape);
                let bytes_out = (q4.blocks.len() * 18) as u64;
                total_bytes_out += bytes_out;

                // Output file: <tensor_name_sanitized>.q4 (replace '.' and '/' with '_')
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
                let out_filename = format!("{sanitized}.q4");
                let out_path = output_dir.join(&out_filename);

                if !dry_run {
                    save_q4_file(&out_path, &q4).unwrap_or_else(|e| {
                        panic!("failed to write {}: {e}", out_path.display());
                    });
                }

                let elapsed = tensor_start.elapsed();
                eprintln!(
                    "  [{}/{n_in_shard}] Q4_0  {tensor_name}  shape={shape:?}  \
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
                // Keep as f16: convert BF16→f32→f16 (or F16→f16 as-is, or F32→f16).
                let f16_data: Vec<u8> = match h.dtype {
                    DType::BF16 => {
                        // BF16 → f32 → f16 (recompress to f16 for uniformity)
                        raw_bytes
                            .chunks_exact(2)
                            .flat_map(|c| {
                                let bf = u16::from_le_bytes([c[0], c[1]]);
                                let f = bf16_to_f32(bf);
                                f32_to_f16(f).to_le_bytes()
                            })
                            .collect()
                    }
                    DType::F16 => {
                        // Already f16: pass through byte-for-byte.
                        raw_bytes.to_vec()
                    }
                    DType::F32 => {
                        // F32 → f16 (lossy downcast for norm weights etc.)
                        raw_bytes
                            .chunks_exact(4)
                            .flat_map(|c| {
                                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                                f32_to_f16(f).to_le_bytes()
                            })
                            .collect()
                    }
                };

                let bytes_out = f16_data.len() as u64;
                total_bytes_out += bytes_out;

                // Output file: <sanitized>.f16 for kept tensors
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
                let out_filename = format!("{sanitized}.f16");
                let out_path = output_dir.join(&out_filename);

                if !dry_run {
                    let mut f = fs::File::create(&out_path).unwrap_or_else(|e| {
                        panic!("failed to create {}: {e}", out_path.display());
                    });
                    use std::io::Write;
                    // Write a minimal header: magic "KHF1" + version u32 + ndim u32 + shape[i] u64 + data
                    f.write_all(b"KHF1").unwrap();
                    f.write_all(&1u32.to_le_bytes()).unwrap();
                    f.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
                    for &dim in shape {
                        f.write_all(&(dim as u64).to_le_bytes()).unwrap();
                    }
                    f.write_all(&(numel as u64).to_le_bytes()).unwrap();
                    f.write_all(&f16_data).unwrap();
                }

                let elapsed = tensor_start.elapsed();
                eprintln!(
                    "  [{}/{n_in_shard}] F16   {tensor_name}  shape={shape:?}  \
                     {:.1}MB  dtype={}  {:.3}s",
                    tensor_idx + 1,
                    bytes_in as f64 / 1_048_576.0,
                    h.dtype.name(),
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

        let shard_elapsed = shard_start.elapsed();
        eprintln!("  Shard done in {:.1}s\n", shard_elapsed.as_secs_f64());

        // Drop mmap explicitly before opening the next shard.
        drop(mmap);
    }

    // Write the quantization index.
    if !dry_run {
        let index_path = output_dir.join("quantize_index.json");
        let index_json = serde_json::to_string_pretty(&index_entries)
            .unwrap_or_else(|e| panic!("failed to serialize index: {e}"));
        fs::write(&index_path, index_json).unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", index_path.display());
        });
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
}
