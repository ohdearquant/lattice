// ---------------------------------------------------------------------------
// doctor subcommand: memory-fit + artifact-compatibility preflight
//
// `lattice doctor <model>` answers "will this load, and what context length
// fits" before any tensor payload is read. It reuses the same tensor-name
// requirements (`qwen_required_tensor_names`) and KV-cache formula
// (`Qwen35Config::kv_bytes_per_token`) the real loaders and the Metal
// forward pass already use, so its numbers describe the actual load path
// rather than a separate approximation. It never touches
// the Metal KV-cache allocator or `new_session`/`new_session_inner`
// (`forward/metal_qwen35/inner/engine.rs`)
// — only their inputs (`Qwen35Config`) and already-published formula.
// ---------------------------------------------------------------------------

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use lattice_arch::qwen35_tensors::{
    FLAT_MODEL_EMBED, LANGUAGE_MODEL_EMBED, inspect_q4_embedding_anchor, q4_tensor_filename,
    qwen_required_tensor_names_with_prefix, resolve_inventory_tensor_prefix,
};
use lattice_inference::model::qwen35_config::Qwen35Config;
use lattice_weights::mmap_trust::{TrustedMmapFile, open_trusted_mmap_file};

/// Bytes per KV-cache element `MetalQwen35Engine::new_session` would use:
/// f32 (4 bytes) unless `LATTICE_KV_F16=1`/`true` — matches that
/// function's own `use_kv_f16` check exactly
/// (`forward/metal_qwen35/inner/engine.rs`).
fn kv_cache_dtype_bytes() -> usize {
    if matches!(
        std::env::var("LATTICE_KV_F16").as_deref(),
        Ok("1") | Ok("true")
    ) {
        2
    } else {
        4
    }
}

/// Which backend a format runs on in *this* binary. Mirrors the
/// dispatch already in `run_chat`/`main`: `Safetensors` always loads via
/// `Qwen35Model::from_safetensors` (CPU); `Q4` always requires the Metal
/// forward pass (`MetalChatBackend` / the serving command's Metal loader).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Cpu,
    Metal,
}

impl std::fmt::Display for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Placement::Cpu => write!(f, "CPU"),
            Placement::Metal => write!(f, "Metal GPU"),
        }
    }
}

/// One discovered tensor: its dtype label (safetensors) and its on-disk
/// byte length.
#[derive(Debug)]
struct TensorEntry {
    dtype: String,
    byte_len: u64,
}

/// Everything discovered about a model directory's weight files,
/// without reading any tensor payload.
struct WeightInventory {
    total_bytes: u64,
    tensor_count: usize,
    quantization: String,
    /// `Some` when the quantization scheme itself could not be read
    /// (e.g. a legacy Q4 v1 file) — a blocking, actionable reason.
    quantization_error: Option<String>,
    /// Tensor names `qwen_required_tensor_names` expects that were not
    /// found on disk.
    missing_tensors: Vec<String>,
    /// Required tensors that exist but use a dtype the loader does not
    /// support.
    unsupported_dtypes: Vec<String>,
    warnings: Vec<String>,
    /// True when the directory contains any `mtp.*` / `mtp_*` tensor
    /// file. Always `false` for the safetensors/CPU path -- MTP is a
    /// Q4/Metal-only feature (`from_q4_dir`'s `load_mtp_q4_weights`).
    /// Q4/Metal directories that load MTP weights allocate a separate
    /// `MetalMtpCache` K/V buffer pair that `kv_bytes_per_token` does
    /// not account for, so this flag drives an explicit disclosure in
    /// `DoctorReport`'s `Display` rather than a silent gap.
    has_mtp_tensors: bool,
}

// -----------------------------------------------------------------------
// pure math: no I/O, directly unit-testable
// -----------------------------------------------------------------------

/// Memory-fit computation. Pure function of already-known inputs; the
/// only formula here is the one described in the issue: KV-cache bytes
/// scale linearly with context length, weight bytes are fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub available_memory_bytes: u64,
    /// Context length the available memory alone allows, ignoring the
    /// model's own `max_position_embeddings` ceiling.
    pub max_context_by_memory: u64,
    pub max_position_embeddings: usize,
    /// `min(max_context_by_memory, max_position_embeddings)` — the
    /// actually-usable maximum.
    pub max_context_len: usize,
    pub requested_context: Option<usize>,
    pub requested_fits: Option<bool>,
}

pub fn plan_memory(
    weight_bytes: u64,
    kv_bytes_per_token: u64,
    available_memory_bytes: u64,
    max_position_embeddings: usize,
    requested_context: Option<usize>,
) -> MemoryPlan {
    let usable = available_memory_bytes.saturating_sub(weight_bytes);
    let max_context_by_memory = if kv_bytes_per_token == 0 {
        u64::MAX
    } else {
        usable / kv_bytes_per_token
    };
    let max_context_len = max_context_by_memory.min(max_position_embeddings as u64) as usize;
    let requested_fits = requested_context.map(|c| c <= max_context_len);
    MemoryPlan {
        weight_bytes,
        kv_bytes_per_token,
        available_memory_bytes,
        max_context_by_memory,
        max_position_embeddings,
        max_context_len,
        requested_context,
        requested_fits,
    }
}

/// `Placement::Metal`'s actual runtime path (`MetalChatBackend`, see its
/// `MAX_CACHE_LEN` doc comment) hard-caps the KV cache at 4096 tokens
/// regardless of `max_position_embeddings` -- without this, doctor
/// could report a context length the CLI's own chat/serve commands
/// would never actually allow. `MetalChatBackend` itself is
/// `metal-gpu`-only, and `doctor` must build without that feature too,
/// so the value is mirrored here (matching the existing
/// `load_q4_config` fallback in `build_report`) rather than shared
/// across the cfg-gate.
pub const METAL_RUNTIME_MAX_CACHE_LEN: usize = 4096;

/// The `max_position_embeddings` value doctor should actually plan
/// against: the model's own architectural ceiling, further capped by
/// [`METAL_RUNTIME_MAX_CACHE_LEN`] for `Placement::Metal` (the
/// CPU/safetensors path has no equivalent runtime cap).
pub fn effective_max_position_embeddings(
    placement: Placement,
    max_position_embeddings: usize,
) -> usize {
    if placement == Placement::Metal {
        max_position_embeddings.min(METAL_RUNTIME_MAX_CACHE_LEN)
    } else {
        max_position_embeddings
    }
}

/// Render a byte count as a human-readable size (KiB/MiB/GiB).
fn human_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

// -----------------------------------------------------------------------
// system memory detection
// -----------------------------------------------------------------------

/// Total physical memory in bytes, or `None` when it cannot be
/// determined (unsupported OS, or the query failed). On Apple Silicon
/// this doubles as the Metal ("VRAM") ceiling: Metal uses unified
/// memory, there is no separate GPU memory pool.
///
/// Mirrors the established `sysctl`-via-`Command` convention already
/// used for system queries in this workspace (`examples/bench_suite.rs`
/// `detect_device`): no new dependency, never panics, degrades to
/// `None` on any failure.
pub fn detect_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    let rest = line.strip_prefix("MemTotal:")?;
                    let kb_str = rest.trim().strip_suffix(" kB")?.trim();
                    kb_str.parse::<u64>().ok().map(|kb| kb * 1024)
                })
            })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

// -----------------------------------------------------------------------
// safetensors weight inventory
// -----------------------------------------------------------------------

/// Read only a safetensors file's JSON header (the 8-byte little-endian
/// length prefix, then that many header bytes) and return each tensor's
/// dtype label and on-disk byte length. Never reads tensor payload
/// bytes. Mirrors the parser already hand-rolled in `quantize_q4.rs`
/// (same `dtype`/`shape`/`data_offsets` keys, same `__metadata__` skip)
/// — this crate has no public API that exposes per-tensor dtype/byte
/// size without either private internals or full tensor materialization.
fn read_safetensors_header(path: &Path) -> Result<HashMap<String, TensorEntry>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
        .len();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| format!("failed to read header length from {}: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > file_len.saturating_sub(8) {
        return Err(format!(
            "{}: header length {header_len} exceeds file size {file_len}",
            path.display()
        ));
    }
    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf)
        .map_err(|e| format!("failed to read header from {}: {e}", path.display()))?;
    let header_str = std::str::from_utf8(&header_buf)
        .map_err(|e| format!("{} header is not valid UTF-8: {e}", path.display()))?;
    let root: serde_json::Value = serde_json::from_str(header_str)
        .map_err(|e| format!("{} header is not valid JSON: {e}", path.display()))?;
    let obj = root
        .as_object()
        .ok_or_else(|| format!("{} header is not a JSON object", path.display()))?;

    let mut out = HashMap::with_capacity(obj.len());
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let Some(offsets) = entry.get("data_offsets").and_then(|v| v.as_array()) else {
            return Err(format!(
                "tensor '{name}' in {} has no data_offsets",
                path.display()
            ));
        };
        if offsets.len() != 2 {
            return Err(format!(
                "tensor '{name}' in {} has malformed data_offsets",
                path.display()
            ));
        }
        let start = offsets[0].as_u64().unwrap_or(0);
        let end = offsets[1].as_u64().unwrap_or(0);
        out.insert(
            name.clone(),
            TensorEntry {
                dtype,
                byte_len: end.saturating_sub(start),
            },
        );
    }
    Ok(out)
}

/// Supported safetensors dtypes — matches `f32_weights.rs`'s private
/// `DType` enum (`F32`, `F16`, `BF16`). Any other dtype on a *required*
/// tensor is reported as an actionable, unsupported-dtype reason rather
/// than discovered only when the real loader fails.
const SUPPORTED_DTYPES: [&str; 3] = ["F32", "F16", "BF16"];

/// Inventory a safetensors model directory: total tensor payload bytes,
/// distinct dtypes observed, and any tensor `qwen_required_tensor_names`
/// expects that is missing or has an unsupported dtype.
///
/// Mirrors `Qwen35Model::from_safetensors`'s own precedence exactly:
/// `model.safetensors` (single file) is preferred over
/// `model.safetensors.index.json` (sharded) when both exist.
/// Bytes one tensor occupies once resident in CPU RAM after
/// `Qwen35Model::from_safetensors` loads it. The CPU path
/// (`crates/weights/src/f32_weights.rs`) always materializes
/// weights as owned `f32`, converting on load via
/// `convert_f16_bytes_to_f32`/`convert_bf16_bytes_to_f32` — so an
/// F16/BF16 tensor's on-disk byte length under-counts its resident
/// footprint by 2x. Dtypes the loader doesn't specially convert are
/// left at their on-disk size; a required tensor in one of those
/// (e.g. an unsupported dtype) is already flagged via
/// `unsupported_dtypes` and blocks a "ready" verdict regardless, so
/// its exact resident size doesn't matter for the feasibility number.
fn cpu_resident_bytes(dtype: &str, on_disk_byte_len: u64) -> u64 {
    match dtype {
        "F16" | "BF16" => on_disk_byte_len * 2,
        _ => on_disk_byte_len,
    }
}

fn required_inventory_tensor_names<'a>(
    cfg: &Qwen35Config,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<String>, String> {
    let prefix = resolve_inventory_tensor_prefix(tensor_names).map_err(|e| e.to_string())?;
    Ok(qwen_required_tensor_names_with_prefix(cfg, prefix))
}

fn inspect_safetensors_dir(dir: &Path, cfg: &Qwen35Config) -> Result<WeightInventory, String> {
    let single = dir.join("model.safetensors");
    let index_path = dir.join("model.safetensors.index.json");

    let all_tensors: HashMap<String, TensorEntry> = if single.exists() {
        read_safetensors_header(&single)?
    } else if index_path.exists() {
        let index_bytes = std::fs::read(&index_path)
            .map_err(|e| format!("failed to read {}: {e}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_slice(&index_bytes)
            .map_err(|e| format!("{} is not valid JSON: {e}", index_path.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("{} has no weight_map object", index_path.display()))?;

        // Dedupe shard filenames — many tensors share one shard.
        let mut shard_names: BTreeSet<String> = BTreeSet::new();
        for v in weight_map.values() {
            if let Some(s) = v.as_str() {
                shard_names.insert(s.to_string());
            }
        }
        let mut merged = HashMap::new();
        for shard_name in shard_names {
            // Index-declared shard names are untrusted checkpoint content;
            // resolve once through the shared containment helper, then
            // report missing files against the resolved path (#1069).
            let shard_path = lattice_weights::contained_shard_path(dir, &shard_name)
                .map_err(|e| e.to_string())?;
            if !shard_path.exists() {
                return Err(format!(
                    "shard '{shard_name}' referenced by {} not found in {}",
                    index_path.display(),
                    dir.display()
                ));
            }
            merged.extend(read_safetensors_header(&shard_path)?);
        }
        merged
    } else {
        return Err(format!(
            "no model.safetensors or model.safetensors.index.json in {}",
            dir.display()
        ));
    };

    let total_bytes: u64 = all_tensors
        .values()
        .map(|t| cpu_resident_bytes(&t.dtype, t.byte_len))
        .sum();
    let dtypes: BTreeSet<&str> = all_tensors.values().map(|t| t.dtype.as_str()).collect();
    let quantization = if dtypes.is_empty() {
        "unknown".to_string()
    } else {
        dtypes.into_iter().collect::<Vec<_>>().join(", ")
    };

    let mut missing_tensors = Vec::new();
    let mut unsupported_dtypes = Vec::new();
    for name in required_inventory_tensor_names(cfg, all_tensors.keys().map(String::as_str))? {
        match all_tensors.get(&name) {
            Some(entry) if !SUPPORTED_DTYPES.contains(&entry.dtype.as_str()) => {
                unsupported_dtypes.push(format!(
                        "tensor '{name}' has dtype {}, which is not supported (supported: F32, F16, BF16)",
                        entry.dtype
                    ));
            }
            Some(_) => {}
            None => missing_tensors.push(name),
        }
    }

    Ok(WeightInventory {
        total_bytes,
        tensor_count: all_tensors.len(),
        quantization,
        quantization_error: None,
        missing_tensors,
        unsupported_dtypes,
        warnings: Vec::new(),
        has_mtp_tensors: false,
    })
}

// -----------------------------------------------------------------------
// Q4 directory weight inventory
// -----------------------------------------------------------------------

/// Estimate a Q4-checkpoint tensor's RESIDENT byte footprint once loaded
/// by `MetalQwen35State::from_q4_dir`
/// (`crates/inference/src/forward/metal_qwen35/inner/weights.rs`),
/// given its on-disk byte length. Mirrors `cpu_resident_bytes` above for
/// the Q4/Metal path: on-disk bytes alone are not a safe RAM/VRAM proxy
/// here either, for two independent reasons — dequantization expansion
/// and runtime-cache duplication:
///
/// - `*.norm.weight` / `A_log` / `dt_bias` / `conv1d.weight` / MTP's
///   `pre_fc_norm_embedding.weight` / `pre_fc_norm_hidden.weight`:
///   `.f16` on disk, loaded via `load_f16_buf_f32` into an f32 Metal
///   buffer — 2x.
/// - `in_proj_a` / `in_proj_b`: Q4 on disk, dequantized to an f16 Metal
///   buffer (`load_q4_as_f16_buf` → `make_buffer_f16_from_q4`). A
///   [`Q4Block`](lattice_weights::q4_weights::Q4Block) packs 32 weights
///   into 20 bytes (0.625 B/elem); f16 resident is 2 B/elem — 3.2x.
/// - `in_proj_qkv` / `in_proj_z`: each is mmap'd zero-copy at its own
///   size (1x) AND its bytes are duplicated again into the merged
///   `in_proj_qkvz` runtime-cache buffer (a mmap'd `merged_qkvz_*.q4`
///   file, or a CPU-concat fallback), which has no manifest/directory
///   entry of its own — so each contributes 2x total.
/// - `embed_tokens`: mmap'd at its on-disk size (1x) for the CPU row
///   lookup. In the *tied* (`tie_word_embeddings == true`) case the GPU
///   logits GEMV maps the same `embed_tokens.weight.q4` file a second
///   time, but both mappings share the file's physical pages, so the
///   pair is still 1x — unlike `in_proj_qkv`/`in_proj_z`, whose second
///   copy is a distinct merged file on disk. In the *untied* case the
///   GPU mapping is the separate `lm_head.weight.q4` file, counted on
///   its own manifest entry, and the embedding stays 1x.
/// - Everything else (full-attention `q/k/v/o_proj`, `mlp.down_proj`,
///   `mlp.gate_proj`/`up_proj` fused by plain concatenation into
///   `gate_up_proj`, `linear_attn.out_proj`, `lm_head`) is mmap'd
///   zero-copy or fused without expansion: resident == on-disk.
///
/// `name_or_file` accepts either the manifest's original dotted tensor
/// name (`quantize_index.json`'s `name` field) or a sanitized
/// `q4_tensor_path`-style filename (dots already replaced with `_`,
/// optionally with a trailing `.q4`/`.f16` extension) — both retain the
/// same distinguishing suffix tokens after normalizing separators.
///
/// This remains an ESTIMATE of peak resident footprint, not a
/// measurement: every mmap counted here is a lazy, file-backed no-copy
/// mapping whose pages fault into physical memory on first touch rather
/// than at map time, so on-disk byte length is an upper bound on eventual
/// residency, not proof of it at any given instant.
fn q4_resident_bytes(name_or_file: &str, on_disk_bytes: u64) -> u64 {
    let mut n = name_or_file.replace('.', "_");
    if let Some(stripped) = n.strip_suffix("_q4").or_else(|| n.strip_suffix("_f16")) {
        n = stripped.to_string();
    }
    if n.ends_with("norm_weight")
        || n.ends_with("A_log")
        || n.ends_with("dt_bias")
        || n.ends_with("conv1d_weight")
        || n.ends_with("pre_fc_norm_embedding_weight")
        || n.ends_with("pre_fc_norm_hidden_weight")
    {
        return on_disk_bytes.saturating_mul(2);
    }
    if n.ends_with("in_proj_a_weight") || n.ends_with("in_proj_b_weight") {
        return (on_disk_bytes as f64 * 3.2).round() as u64;
    }
    if n.ends_with("in_proj_qkv_weight") || n.ends_with("in_proj_z_weight") {
        return on_disk_bytes.saturating_mul(2);
    }
    if n.ends_with("embed_tokens_weight") {
        return on_disk_bytes;
    }
    on_disk_bytes
}

fn is_q4_cache_filename(name: &str) -> bool {
    name.starts_with("merged_qkvz_")
}

fn is_non_cache_q4_filename(name: &str) -> bool {
    name.ends_with(".q4") && !is_q4_cache_filename(name)
}

struct Q4QuantizationSample {
    path: PathBuf,
    file: TrustedMmapFile,
}

/// Pick the first sorted non-cache `.q4` candidate accepted by the native loader's
/// trusted opener, retaining its opened handle for header inspection.
///
/// The shared opener rejects final-component symlinks on Unix and applies its
/// platform-specific file-kind and trust checks. Every refusal is retained while
/// later candidates are tried; the first refusal is returned only if none succeeds.
/// Returns `Ok(None)` only when there are no eligible filenames.
fn select_q4_quantization_sample(dir: &Path) -> Result<Option<Q4QuantizationSample>, String> {
    select_q4_quantization_sample_with_opener(dir, open_trusted_mmap_file)
}

fn select_q4_quantization_sample_with_opener(
    dir: &Path,
    mut open_file: impl FnMut(&Path) -> Result<TrustedMmapFile, String>,
) -> Result<Option<Q4QuantizationSample>, String> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read directory {}: {e}", dir.display()))?
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .map(is_non_cache_q4_filename)
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();
    candidates.sort();
    let mut first_error = None;
    for path in candidates {
        match open_file(&path) {
            Ok(file) => return Ok(Some(Q4QuantizationSample { path, file })),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(None),
    }
}

/// Sample the header of the file `select_q4_quantization_sample` picks to identify the
/// quantization format, using the real, already-shipped
/// `read_q4_header` (a header-only read — no block payload is decoded).
/// Catches a legacy v1 file or other corruption the same way the Metal
/// loader would.
fn detect_q4_quantization_label(dir: &Path) -> Result<String, String> {
    let Some(sample) = select_q4_quantization_sample(dir)? else {
        return Ok("Q4 (no .q4 files found to sample)".to_string());
    };
    q4_quantization_label_from_sample(sample)
}

fn q4_quantization_label_from_sample(sample: Q4QuantizationSample) -> Result<String, String> {
    let Q4QuantizationSample { path, mut file } = sample;
    lattice_weights::q4_weights::read_q4_header(&mut file)
        .map(|_| "Q4_0 (lattice native, v2 asymmetric)".to_string())
        .map_err(|e| format!("unsupported quantization scheme in {}: {e}", path.display()))
}

fn inspect_q4_manifest_embedding_anchor(
    dir: &Path,
    tensor_name: &str,
    manifest_file: &str,
    contained_manifest_path: &Path,
) -> (Result<std::fs::Metadata, String>, Option<String>) {
    // Native loading opens fixed embedding filenames, not manifest aliases.
    let expected_file = q4_tensor_filename(tensor_name, "q4");
    let native_path = dir.join(&expected_file);
    let is_alias = contained_manifest_path != native_path.as_path();
    let warning = if is_alias {
        Some(format!(
            "{tensor_name}: manifest alias '{manifest_file}' ignored; the loader reads '{expected_file}'"
        ))
    } else {
        None
    };
    let metadata = match inspect_q4_embedding_anchor(&native_path) {
        Ok(Some(metadata)) => Ok(metadata),
        Ok(None) if is_alias => Err(format!("native embedding file '{expected_file}' not found")),
        Ok(None) => Err("file not found".to_string()),
        Err(error) => Err(error.to_string()),
    };
    (metadata, warning)
}

/// Inventory a native Q4 quantized directory (the output of
/// `quantize_q4`). Prefers `quantize_index.json` — the manifest
/// `quantize_q4` writes listing exactly the original per-tensor
/// `.q4`/`.f16` files — over a raw directory scan. This matters: on
/// first Metal load, `MetalQwen35State` creates `merged_qkvz_*.q4`
/// runtime-cache files that merge (and duplicate the bytes of) each
/// layer's still-present `in_proj_qkv`/`in_proj_z` source tensors: a
/// directory scan that does not exclude them double-counts. The
/// manifest sidesteps the problem entirely since it lists only the
/// original tensors; the fallback path (no manifest) excludes any
/// `merged_qkvz_`-prefixed file explicitly.
fn inspect_q4_dir(dir: &Path, cfg: &Qwen35Config) -> Result<WeightInventory, String> {
    let (quantization, quantization_error) = match detect_q4_quantization_label(dir) {
        Ok(label) => (label, None),
        Err(e) => ("unknown (see blocking reasons)".to_string(), Some(e)),
    };

    // `quantize_index.json` parsing/validation (bounded read + shape
    // normalization across both writer flavors) is centralized in
    // `lattice_inference::quant::q4_manifest` (issue #655); `doctor`
    // only inventories tensors, so the QuaRot seed field is unused here.
    let manifest = lattice_inference::quant::q4_manifest::load_manifest(dir)?;
    if let Some(manifest) = manifest {
        let entries = manifest.tensors;

        let mut total_bytes = 0u64;
        let mut missing_tensors = Vec::new();
        let mut warnings = Vec::new();
        let mut present_names: BTreeSet<String> = BTreeSet::new();
        let mut has_mtp_tensors = false;
        for entry in &entries {
            // Manifest-declared file names are untrusted checkpoint
            // content; the same lexical containment the loaders apply
            // gates the doctor's readiness accounting, so an entry the
            // loader would reject is never counted present (#1069).
            let Ok(file_path) = lattice_weights::contained_shard_path(dir, &entry.file) else {
                missing_tensors.push(format!(
                    "{} (listed in quantize_index.json as '{}', \
                         path escapes the model directory)",
                    entry.name, entry.file
                ));
                continue;
            };
            let metadata = if matches!(entry.name.as_str(), LANGUAGE_MODEL_EMBED | FLAT_MODEL_EMBED)
            {
                let (metadata, warning) =
                    inspect_q4_manifest_embedding_anchor(dir, &entry.name, &entry.file, &file_path);
                if let Some(warning) = warning {
                    warnings.push(warning);
                }
                metadata
            } else {
                std::fs::metadata(&file_path).map_err(|_| "file not found".to_string())
            };
            match metadata {
                Ok(meta) => {
                    total_bytes += q4_resident_bytes(&entry.name, meta.len());
                    present_names.insert(entry.name.clone());
                    has_mtp_tensors |= entry.name.starts_with("mtp.");
                }
                Err(error) => missing_tensors.push(format!(
                    "{} (listed in quantize_index.json as '{}', {error})",
                    entry.name, entry.file
                )),
            }
        }
        for name in required_inventory_tensor_names(cfg, present_names.iter().map(String::as_str))?
        {
            if !present_names.contains(&name) {
                missing_tensors.push(name);
            }
        }

        Ok(WeightInventory {
            total_bytes,
            tensor_count: entries.len(),
            quantization,
            quantization_error,
            missing_tensors,
            unsupported_dtypes: Vec::new(),
            warnings,
            has_mtp_tensors,
        })
    } else {
        // No manifest: fall back to a directory scan, excluding
        // merged_qkvz_* runtime-cache files (see doc comment above).
        // Tensor-name coverage is not checked in this path — sanitized
        // filenames don't reliably reverse to the original dotted
        // tensor names, so `missing_tensors` is intentionally left
        // empty here rather than guessed.
        let mut total_bytes = 0u64;
        let mut tensor_count = 0usize;
        let mut has_mtp_tensors = false;
        let read_dir = std::fs::read_dir(dir)
            .map_err(|e| format!("failed to read directory {}: {e}", dir.display()))?;
        for entry in read_dir.flatten() {
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            if (is_non_cache_q4_filename(name)
                || (name.ends_with(".f16") && !is_q4_cache_filename(name)))
                && let Ok(meta) = entry.metadata()
            {
                total_bytes += q4_resident_bytes(name, meta.len());
                tensor_count += 1;
                has_mtp_tensors |= name.starts_with("mtp_");
            }
        }
        Ok(WeightInventory {
            total_bytes,
            tensor_count,
            quantization,
            quantization_error,
            missing_tensors: Vec::new(),
            unsupported_dtypes: Vec::new(),
            warnings: Vec::new(),
            has_mtp_tensors,
        })
    }
}

// -----------------------------------------------------------------------
// top-level report
// -----------------------------------------------------------------------

/// Full preflight report for one model directory.
#[derive(Debug)]
pub struct DoctorReport {
    pub model_dir: PathBuf,
    pub format: crate::backend::ModelFormat,
    pub placement: Placement,
    pub quantization: String,
    pub tensor_count: usize,
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub max_position_embeddings: usize,
    /// `Some(4096)` for `Placement::Metal` -- the actual runtime cap
    /// `MetalChatBackend::MAX_CACHE_LEN` imposes regardless of the
    /// model's own `max_position_embeddings`. `None` for `Placement::Cpu`,
    /// which has no equivalent hard cap.
    pub metal_runtime_cache_cap: Option<usize>,
    pub available_memory_bytes: Option<u64>,
    pub max_context_len: Option<usize>,
    pub requested_context: Option<usize>,
    pub requested_fits: Option<bool>,
    pub tokenizer_path: PathBuf,
    pub tokenizer_present: bool,
    pub missing_tensors: Vec<String>,
    /// Non-empty ⇒ this artifact is not ready to run as configured.
    /// Each entry is a standalone, actionable explanation.
    pub blocking_reasons: Vec<String>,
    /// Non-blocking inventory notices shown alongside readiness diagnostics.
    pub warnings: Vec<String>,
    /// True when the Q4 directory has MTP tensor files -- `kv_bytes_per_token`
    /// above only ever covers the main model's full-attention KV cache,
    /// never the separate `MetalMtpCache` K/V buffers `from_q4_dir`
    /// allocates when MTP weights load, so this drives an explicit
    /// disclosure line rather than a silently-optimistic context estimate.
    pub has_mtp_tensors: bool,
}

impl DoctorReport {
    pub fn is_ready(&self) -> bool {
        self.blocking_reasons.is_empty()
    }
}

fn write_report_diagnostic(
    f: &mut std::fmt::Formatter<'_>,
    prefix: &str,
    text: &str,
) -> std::fmt::Result {
    f.write_str(prefix)?;
    for character in text.chars() {
        if character.is_control() {
            write!(f, "{}", character.escape_default())?;
        } else {
            write!(f, "{character}")?;
        }
    }
    writeln!(f)
}

impl std::fmt::Display for DoctorReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model directory : {}", self.model_dir.display())?;
        writeln!(f, "Format          : {:?}", self.format)?;
        writeln!(f, "Placement       : {}", self.placement)?;
        writeln!(f, "Quantization    : {}", self.quantization)?;
        writeln!(f, "Tensors found   : {}", self.tensor_count)?;
        writeln!(
            f,
            "Weight memory   : {} ({} bytes)",
            human_bytes(self.weight_bytes),
            self.weight_bytes
        )?;
        writeln!(
            f,
            "KV cache        : {}/token ({} bytes/token; at 4096 tokens ~= {})",
            human_bytes(self.kv_bytes_per_token),
            self.kv_bytes_per_token,
            human_bytes(self.kv_bytes_per_token.saturating_mul(4096))
        )?;
        writeln!(
            f,
            "Model max context (max_position_embeddings): {}",
            self.max_position_embeddings
        )?;
        if let Some(cap) = self.metal_runtime_cache_cap {
            writeln!(
                f,
                "Metal runtime cache cap: {cap} tokens (MetalChatBackend::MAX_CACHE_LEN; \
                     the chat/serve binaries never allow more, even if memory allows it)"
            )?;
        }
        match self.available_memory_bytes {
            Some(avail) => writeln!(
                f,
                "Detected system memory: {} ({} bytes)",
                human_bytes(avail),
                avail
            )?,
            None => writeln!(
                f,
                "Detected system memory: unknown (unsupported OS or query failed)"
            )?,
        }
        match self.max_context_len {
            Some(max_ctx) => writeln!(
                f,
                "Max feasible context length: {max_ctx} tokens \
                     (weights + KV cache only -- activation buffers, GDN recurrent-state \
                     scratch, and tokenizer tables are not counted)"
            )?,
            None => writeln!(
                f,
                "Max feasible context length: unknown (system memory undetected)"
            )?,
        }
        if self.has_mtp_tensors {
            writeln!(
                f,
                "Note: this directory includes MTP (multi-token prediction) files -- \
                     the separate MTP K/V cache and session buffers `from_q4_dir` allocates \
                     for them are also not counted in the estimate above."
            )?;
        }
        if let Some(requested) = self.requested_context {
            match self.requested_fits {
                Some(true) => writeln!(f, "Requested context {requested}: fits")?,
                Some(false) => writeln!(f, "Requested context {requested}: DOES NOT FIT")?,
                None => writeln!(
                    f,
                    "Requested context {requested}: unknown (system memory undetected)"
                )?,
            }
        }
        writeln!(
            f,
            "Tokenizer       : {} ({})",
            self.tokenizer_path.display(),
            if self.tokenizer_present {
                "found"
            } else {
                "MISSING"
            }
        )?;
        if !self.missing_tensors.is_empty() {
            writeln!(
                f,
                "Missing required tensors ({}):",
                self.missing_tensors.len()
            )?;
            for name in self.missing_tensors.iter().take(20) {
                write_report_diagnostic(f, "  - ", name)?;
            }
            if self.missing_tensors.len() > 20 {
                writeln!(f, "  ... and {} more", self.missing_tensors.len() - 20)?;
            }
        }
        for warning in &self.warnings {
            write_report_diagnostic(f, "Warning: ", warning)?;
        }
        writeln!(f)?;
        if self.is_ready() {
            writeln!(f, "Result: OK -- weights + KV cache fit; ready to load")?;
        } else {
            writeln!(f, "Result: NOT READY")?;
            for reason in &self.blocking_reasons {
                write_report_diagnostic(f, "  - ", reason)?;
            }
        }
        Ok(())
    }
}

/// Build a full preflight report for `model_dir` without loading any
/// tensor payload.
///
/// `available_memory_override` exists so tests can simulate a
/// memory-constrained machine deterministically; real callers (the
/// `doctor` CLI subcommand) always pass `None`, which detects the
/// machine's actual total memory via [`detect_total_memory_bytes`].
pub fn build_report(
    model_dir: &Path,
    tokenizer_dir: Option<&Path>,
    requested_context: Option<usize>,
    available_memory_override: Option<u64>,
) -> Result<DoctorReport, String> {
    let format = crate::backend::detect_format(model_dir);

    let (placement, cfg, inventory) = match format {
        crate::backend::ModelFormat::Safetensors => {
            let cfg = Qwen35Config::from_model_dir(model_dir)
                .map_err(|e| format!("config.json load failed: {e}"))?;
            let inventory = inspect_safetensors_dir(model_dir, &cfg)?;
            (Placement::Cpu, cfg, inventory)
        }
        crate::backend::ModelFormat::Q4 => {
            let cfg = Qwen35Config::from_model_dir(model_dir)
                .map_err(|e| format!("config.json load failed: {e}"))?;
            let inventory = inspect_q4_dir(model_dir, &cfg)?;
            (Placement::Metal, cfg, inventory)
        }
        crate::backend::ModelFormat::Unknown => {
            return Err(crate::backend::unrecognized_format_message(model_dir));
        }
        // Any format this binary doesn't yet know how to inspect is handled
        // the same way as `Unknown`: report it as an error rather than
        // silently guessing an inventory shape.
        _ => {
            return Err(crate::backend::unloadable_format_message(model_dir));
        }
    };

    let tokenizer_path = tokenizer_dir.unwrap_or(model_dir).join("tokenizer.json");
    let tokenizer_present = tokenizer_path.exists();

    let kv_bytes_per_token = cfg.kv_bytes_per_token(kv_cache_dtype_bytes()) as u64;
    let available_memory_bytes = available_memory_override.or_else(detect_total_memory_bytes);

    let effective_max_position_embeddings =
        effective_max_position_embeddings(placement, cfg.max_position_embeddings);

    let (max_context_len, requested_fits) = match available_memory_bytes {
        Some(avail) => {
            let plan = plan_memory(
                inventory.total_bytes,
                kv_bytes_per_token,
                avail,
                effective_max_position_embeddings,
                requested_context,
            );
            (Some(plan.max_context_len), plan.requested_fits)
        }
        None => (None, None),
    };

    let mut blocking_reasons = Vec::new();
    if let Some(e) = &inventory.quantization_error {
        blocking_reasons.push(e.clone());
    }
    blocking_reasons.extend(inventory.unsupported_dtypes.iter().cloned());
    if !inventory.missing_tensors.is_empty() {
        blocking_reasons.push(format!(
            "{} required tensor(s) missing (see list below), e.g. '{}'",
            inventory.missing_tensors.len(),
            inventory.missing_tensors[0]
        ));
    }
    if !tokenizer_present {
        blocking_reasons.push(format!(
            "tokenizer.json not found at {}",
            tokenizer_path.display()
        ));
    }
    if format == crate::backend::ModelFormat::Q4 && !cfg!(feature = "metal-gpu") {
        blocking_reasons.push(crate::backend::metal_gpu_required_message(model_dir));
    }
    if requested_fits == Some(false) {
        let requested = requested_context.unwrap_or(0);
        let max_ctx = max_context_len.unwrap_or(0);
        blocking_reasons.push(format!(
            "requested context {requested} does not fit: max feasible is {max_ctx} tokens"
        ));
    }
    // Fail closed on the doctor's own computed numbers even when no
    // `--context` was requested (#875): a machine that cannot even fit
    // the weights, or that fits weights but leaves zero room for a
    // single token of KV cache, is not "ready to load" regardless of
    // whether the caller asked about a specific context length.
    if let Some(avail) = available_memory_bytes
        && inventory.total_bytes > avail
    {
        blocking_reasons.push(format!(
            "estimated weight memory {} exceeds detected system memory {}: this model is \
                 unlikely to load on this machine (estimate of peak resident footprint from \
                 on-disk tensor sizes -- mmap'd weights fault into physical memory lazily on \
                 first touch, so this is not a measurement of memory actually held at any \
                 instant)",
            human_bytes(inventory.total_bytes),
            human_bytes(avail)
        ));
    }
    if max_context_len == Some(0) {
        blocking_reasons.push(
            "max feasible context length is 0 tokens: weights and/or KV cache do not fit \
                 in available memory (not even a single token of context)"
                .to_string(),
        );
    }

    Ok(DoctorReport {
        model_dir: model_dir.to_path_buf(),
        format,
        placement,
        quantization: inventory.quantization,
        tensor_count: inventory.tensor_count,
        weight_bytes: inventory.total_bytes,
        kv_bytes_per_token,
        max_position_embeddings: cfg.max_position_embeddings,
        metal_runtime_cache_cap: (placement == Placement::Metal)
            .then_some(METAL_RUNTIME_MAX_CACHE_LEN),
        available_memory_bytes,
        max_context_len,
        requested_context,
        requested_fits,
        tokenizer_path,
        tokenizer_present,
        missing_tensors: inventory.missing_tensors,
        blocking_reasons,
        warnings: inventory.warnings,
        has_mtp_tensors: inventory.has_mtp_tensors,
    })
}

#[cfg(test)]
mod tests;
