// ---------------------------------------------------------------------------
// doctor subcommand: memory-fit + artifact-compatibility preflight
//
// `lattice doctor <model>` answers "will this load, and what context length
// fits" before any tensor payload is read. It reuses the same tensor-name
// requirements (`qwen_required_tensor_names`) and KV-cache formula
// (`Qwen35Config::kv_bytes_per_token`) the real loaders and the Metal
// forward pass already use, so its numbers describe the actual load path
// rather than a separate approximation. It never touches
// `metal_qwen35.rs`'s KV-cache allocator or `new_session`/`new_session_inner`
// — only their inputs (`Qwen35Config`) and already-published formula.
// ---------------------------------------------------------------------------

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use lattice_inference::model::qwen35::qwen_required_tensor_names;
use lattice_inference::model::qwen35_config::Qwen35Config;

/// Bytes per KV-cache element `MetalQwen35State::new_session` would use:
/// f32 (4 bytes) unless `LATTICE_KV_F16=1`/`true` — matches that
/// function's own `use_kv_f16` check exactly (`metal_qwen35.rs`).
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
/// forward pass (`MetalChatBackend` / `serve::ModelBackend::spawn_metal`).
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
/// (`crates/inference/src/weights/f32_weights.rs`) always materializes
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
            let shard_path = lattice_inference::weights::contained_shard_path(dir, &shard_name)
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
    for name in qwen_required_tensor_names(cfg) {
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
        has_mtp_tensors: false,
    })
}

// -----------------------------------------------------------------------
// Q4 directory weight inventory
// -----------------------------------------------------------------------

/// Estimate a Q4-checkpoint tensor's RESIDENT byte footprint once loaded
/// by `MetalQwen35State::from_q4_dir` (`crates/inference/src/forward/metal_qwen35.rs`),
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
///   [`Q4Block`](crate::weights::q4_weights::Q4Block) packs 32 weights
///   into 20 bytes (0.625 B/elem); f16 resident is 2 B/elem — 3.2x.
/// - `in_proj_qkv` / `in_proj_z`: each is mmap'd zero-copy at its own
///   size (1x) AND its bytes are duplicated again into the merged
///   `in_proj_qkvz` runtime-cache buffer (a mmap'd `merged_qkvz_*.q4`
///   file, or a CPU-concat fallback), which has no manifest/directory
///   entry of its own — so each contributes 2x total.
/// - `embed_tokens`: dequantized into a full f16 buffer for the CPU
///   embedding lookup (3.2x, same ratio as `in_proj_a`/`b`) AND, only in
///   the *tied* (`tie_word_embeddings == true`) case, ALSO separately
///   mmap'd at its own on-disk size for the GPU logits GEMV
///   (`embed_tokens_q8`) — 4.2x total for tied checkpoints. In the
///   *untied* case, `MetalQwen35State::from_q4_dir` immediately shadows
///   that same-named local binding with a fresh mmap of the separate
///   `lm_head.weight.q4` file (`metal_qwen35.rs`, the
///   `!cfg.tie_word_embeddings` branch a few lines after the tied
///   assignment) — the embedding's own Q4 mmap is dropped before the
///   function returns, so only the 3.2x f16 dequant survives, and
///   `lm_head.weight` is counted separately (and correctly, as a plain
///   1x mmap) via its own manifest/directory entry. Flattening this to
///   a constant 4.2x regardless of `tie_word_embeddings` was a
///   deliberate simplification when this estimator was pure telemetry,
///   but it over-counts untied checkpoints by one full embedding-sized
///   mmap once the estimate started gating a hard pass/fail verdict
///   (#881) — hence the `tie_word_embeddings` parameter below.
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
/// Even with the `tie_word_embeddings` correction this remains an
/// ESTIMATE of peak resident footprint, not a measurement: every mmap
/// counted here is a lazy, file-backed no-copy mapping whose pages
/// fault into physical memory on first touch rather than at map time,
/// so on-disk byte length is an upper bound on eventual residency, not
/// proof of it at any given instant.
fn q4_resident_bytes(name_or_file: &str, on_disk_bytes: u64, tie_word_embeddings: bool) -> u64 {
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
        let ratio = if tie_word_embeddings { 4.2 } else { 3.2 };
        return (on_disk_bytes as f64 * ratio).round() as u64;
    }
    on_disk_bytes
}

/// Sample one non-cache `.q4` file's header to identify the
/// quantization format, using the real, already-shipped
/// `read_q4_header` (a header-only read — no block payload is decoded).
/// Catches a legacy v1 file or other corruption the same way the Metal
/// loader would.
fn detect_q4_quantization_label(dir: &Path) -> Result<String, String> {
    let sample = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read directory {}: {e}", dir.display()))?
        .flatten()
        .find(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.ends_with(".q4") && !n.starts_with("merged_qkvz_"))
                .unwrap_or(false)
        });
    let Some(sample) = sample else {
        return Ok("Q4 (no .q4 files found to sample)".to_string());
    };
    let mut file = std::fs::File::open(sample.path())
        .map_err(|e| format!("failed to open {}: {e}", sample.path().display()))?;
    lattice_inference::weights::q4_weights::read_q4_header(&mut file)
        .map(|_| "Q4_0 (lattice native, v2 asymmetric)".to_string())
        .map_err(|e| {
            format!(
                "unsupported quantization scheme in {}: {e}",
                sample.path().display()
            )
        })
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
        let mut present_names: BTreeSet<String> = BTreeSet::new();
        let mut has_mtp_tensors = false;
        for entry in &entries {
            // Manifest-declared file names are untrusted checkpoint
            // content; the same lexical containment the loaders apply
            // gates the doctor's readiness accounting, so an entry the
            // loader would reject is never counted present (#1069).
            let Ok(file_path) = lattice_inference::weights::contained_shard_path(dir, &entry.file)
            else {
                missing_tensors.push(format!(
                    "{} (listed in quantize_index.json as '{}', \
                         path escapes the model directory)",
                    entry.name, entry.file
                ));
                continue;
            };
            match std::fs::metadata(&file_path) {
                Ok(meta) => {
                    total_bytes +=
                        q4_resident_bytes(&entry.name, meta.len(), cfg.tie_word_embeddings);
                    present_names.insert(entry.name.clone());
                    has_mtp_tensors |= entry.name.starts_with("mtp.");
                }
                Err(_) => missing_tensors.push(format!(
                    "{} (listed in quantize_index.json as '{}', file not found)",
                    entry.name, entry.file
                )),
            }
        }
        for name in qwen_required_tensor_names(cfg) {
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
            if name.starts_with("merged_qkvz_") {
                continue;
            }
            if (name.ends_with(".q4") || name.ends_with(".f16"))
                && let Ok(meta) = entry.metadata()
            {
                total_bytes += q4_resident_bytes(name, meta.len(), cfg.tie_word_embeddings);
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
                writeln!(f, "  - {name}")?;
            }
            if self.missing_tensors.len() > 20 {
                writeln!(f, "  ... and {} more", self.missing_tensors.len() - 20)?;
            }
        }
        writeln!(f)?;
        if self.is_ready() {
            writeln!(f, "Result: OK -- weights + KV cache fit; ready to load")?;
        } else {
            writeln!(f, "Result: NOT READY")?;
            for reason in &self.blocking_reasons {
                writeln!(f, "  - {reason}")?;
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
        has_mtp_tensors: inventory.has_mtp_tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tempdir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "lattice-doctor-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }

    // ---- human_bytes ----------------------------------------------

    #[test]
    fn human_bytes_formats_units() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.00 KiB");
        assert_eq!(human_bytes(1024 * 1024), "1.00 MiB");
        assert_eq!(human_bytes(1024 * 1024 * 1024), "1.00 GiB");
        assert_eq!(human_bytes(1536 * 1024 * 1024), "1.50 GiB");
    }

    // ---- plan_memory (pure math) ------------------------------------

    #[test]
    fn plan_memory_hand_computed() {
        // weight=100 bytes, kv=10 bytes/token, available=1000 bytes.
        // usable = 1000 - 100 = 900. max_by_memory = 900 / 10 = 90.
        let plan = plan_memory(100, 10, 1000, 1_000_000, Some(50));
        assert_eq!(plan.max_context_by_memory, 90);
        assert_eq!(plan.max_context_len, 90); // min(90, 1_000_000)
        assert_eq!(plan.requested_fits, Some(true)); // 50 <= 90

        let plan2 = plan_memory(100, 10, 1000, 1_000_000, Some(95));
        assert_eq!(plan2.requested_fits, Some(false)); // 95 > 90
    }

    #[test]
    fn plan_memory_capped_by_max_position_embeddings() {
        // Effectively unlimited memory; the model architecture caps
        // context instead.
        let plan = plan_memory(0, 1, 1_000_000_000_000, 4096, None);
        assert_eq!(plan.max_context_by_memory, 1_000_000_000_000);
        assert_eq!(plan.max_context_len, 4096);
    }

    // ---- effective_max_position_embeddings (Medium-2 fix) -----------

    #[test]
    fn effective_max_position_embeddings_caps_metal_placement_at_runtime_limit() {
        // Qwen3.6-27B's real max_position_embeddings (131072) far
        // exceeds MetalChatBackend's actual 4096-token runtime cap --
        // doctor must not report a feasible context the binary would
        // refuse to serve.
        assert_eq!(
            effective_max_position_embeddings(Placement::Metal, 131_072),
            METAL_RUNTIME_MAX_CACHE_LEN
        );
    }

    #[test]
    fn effective_max_position_embeddings_leaves_smaller_model_ceiling_untouched() {
        // A model whose own ceiling is already below the runtime cap
        // must not be inflated up to it.
        assert_eq!(
            effective_max_position_embeddings(Placement::Metal, 2048),
            2048
        );
    }

    #[test]
    fn effective_max_position_embeddings_does_not_cap_cpu_placement() {
        // The CPU/safetensors path has no equivalent runtime cache cap.
        assert_eq!(
            effective_max_position_embeddings(Placement::Cpu, 131_072),
            131_072
        );
    }

    #[test]
    fn plan_memory_weight_bytes_exceeding_available_yields_zero_context() {
        // Weights alone don't fit -- no room for any KV cache at all.
        let plan = plan_memory(2_000, 10, 1_000, 1_000_000, Some(1));
        assert_eq!(plan.max_context_by_memory, 0);
        assert_eq!(plan.max_context_len, 0);
        assert_eq!(plan.requested_fits, Some(false));
    }

    #[test]
    fn plan_memory_kv_bytes_per_token_matches_qwen35_0_8b_doc_identity() {
        // Cross-check against `Qwen35Config::kv_bytes_per_token`'s own
        // doc-comment worked example: 6 full-attention layers * 2 (K+V)
        // * 512 full_kv_dim * 2 bytes (f16) = 12_288 bytes/token.
        let cfg = Qwen35Config::qwen35_0_8b();
        assert_eq!(cfg.num_full_attention_layers(), 6);
        assert_eq!(cfg.full_kv_dim(), 512);
        assert_eq!(cfg.kv_bytes_per_token(2), 12_288);
        assert_eq!(cfg.kv_bytes_per_token(4), 24_576);
    }

    // ---- safetensors header parsing ---------------------------------

    fn write_fake_safetensors(path: &Path, tensors: &[(&str, &str, u64, u64)]) {
        let mut header = serde_json::Map::new();
        for (name, dtype, start, end) in tensors {
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": [1],
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_json = serde_json::Value::Object(header).to_string();
        let header_bytes = header_json.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        let payload_len = tensors.iter().map(|(_, _, _, e)| *e).max().unwrap_or(0);
        buf.resize(buf.len() + payload_len as usize, 0);
        fs::write(path, buf).expect("write fake safetensors file");
    }

    #[test]
    fn read_safetensors_header_computes_exact_byte_lengths() {
        let dir = tempdir("st-header");
        let path = dir.join("model.safetensors");
        write_fake_safetensors(
            &path,
            &[("tensor.a", "F32", 0, 400), ("tensor.b", "BF16", 400, 600)],
        );
        let tensors = read_safetensors_header(&path).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["tensor.a"].byte_len, 400);
        assert_eq!(tensors["tensor.a"].dtype, "F32");
        assert_eq!(tensors["tensor.b"].byte_len, 200);
        assert_eq!(tensors["tensor.b"].dtype, "BF16");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_safetensors_header_rejects_oversized_header_length() {
        let dir = tempdir("st-header-bad");
        let path = dir.join("model.safetensors");
        // A header-length prefix claiming far more bytes than the file
        // actually has -- must fail, never attempt the allocation.
        let mut buf = Vec::new();
        buf.extend_from_slice(&(1_000_000_000u64).to_le_bytes());
        buf.extend_from_slice(b"tiny");
        fs::write(&path, buf).unwrap();
        let err = read_safetensors_header(&path).unwrap_err();
        assert!(err.contains("exceeds file size"));
        fs::remove_dir_all(&dir).ok();
    }

    fn required_tensor_fixture(cfg: &Qwen35Config) -> Vec<(String, String, u64, u64)> {
        let mut offset = 0u64;
        qwen_required_tensor_names(cfg)
            .into_iter()
            .map(|name| {
                let start = offset;
                offset += 64;
                (name, "F32".to_string(), start, offset)
            })
            .collect()
    }

    #[test]
    fn inspect_safetensors_dir_all_tensors_present_no_missing() {
        let dir = tempdir("st-complete");
        let cfg = Qwen35Config::qwen35_0_8b();
        let tensors = required_tensor_fixture(&cfg);
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);

        let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
        assert!(inv.missing_tensors.is_empty());
        assert!(inv.unsupported_dtypes.is_empty());
        assert_eq!(inv.total_bytes, tensors.len() as u64 * 64);
        assert_eq!(inv.quantization, "F32");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_safetensors_dir_scales_f16_bf16_to_resident_f32_bytes() {
        // The CPU loader always materializes weights as owned f32,
        // converting F16/BF16 on load (see `cpu_resident_bytes`'s doc
        // comment) -- so a 2-byte-per-element on-disk tensor must count
        // double toward the RAM budget, while F32 tensors (already 4
        // bytes/elem) must not be scaled.
        let dir = tempdir("st-mixed-dtype");
        let cfg = Qwen35Config::qwen35_0_8b();
        let mut tensors = required_tensor_fixture(&cfg);
        assert!(
            tensors.len() >= 2,
            "fixture must have room to mutate two entries"
        );
        tensors[0].1 = "F16".to_string();
        tensors[1].1 = "BF16".to_string();
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);

        let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
        // Every tensor is 64 on-disk bytes; the two F16/BF16 entries
        // must count as 128 resident bytes each, the rest stay at 64.
        let expected = (tensors.len() as u64 - 2) * 64 + 2 * 128;
        assert_eq!(inv.total_bytes, expected);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_safetensors_dir_detects_missing_tensor() {
        let dir = tempdir("st-missing");
        let cfg = Qwen35Config::qwen35_0_8b();
        let mut tensors = required_tensor_fixture(&cfg);
        let (dropped_name, _, _, _) = tensors.pop().unwrap();
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);

        let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
        assert_eq!(inv.missing_tensors, vec![dropped_name]);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_safetensors_dir_flags_unsupported_dtype() {
        let dir = tempdir("st-baddtype");
        let cfg = Qwen35Config::qwen35_0_8b();
        let mut tensors = required_tensor_fixture(&cfg);
        // Corrupt one required tensor's dtype to something unsupported.
        tensors[0].1 = "I64".to_string();
        let bad_name = tensors[0].0.clone();
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);

        let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
        assert_eq!(inv.unsupported_dtypes.len(), 1);
        assert!(inv.unsupported_dtypes[0].contains(&bad_name));
        assert!(inv.unsupported_dtypes[0].contains("I64"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_safetensors_dir_prefers_single_file_over_index() {
        // Mirrors `Qwen35Model::from_safetensors`'s precedence: when
        // both `model.safetensors` and `model.safetensors.index.json`
        // exist, the single file wins and the index is never consulted.
        let dir = tempdir("st-precedence");
        let cfg = Qwen35Config::qwen35_0_8b();
        let tensors = required_tensor_fixture(&cfg);
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);
        // A bogus index.json that would error if it were ever read.
        fs::write(dir.join("model.safetensors.index.json"), b"not valid json").unwrap();

        let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
        assert!(inv.missing_tensors.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    // ---- Q4 directory inventory --------------------------------------

    fn write_fake_q4_file(
        path: &Path,
        version: u32,
        shape: &[u64],
        original_len: u64,
        n_blocks: usize,
    ) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHQ4");
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for s in shape {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf.extend_from_slice(&original_len.to_le_bytes());
        buf.resize(buf.len() + n_blocks * 20, 0);
        fs::write(path, buf).expect("write fake q4 file");
    }

    #[test]
    fn detect_q4_quantization_label_accepts_v2_file() {
        let dir = tempdir("q4-label-v2");
        write_fake_q4_file(&dir.join("sample.q4"), 2, &[32], 32, 1);
        let label = detect_q4_quantization_label(&dir).unwrap();
        assert!(label.contains("Q4_0"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_q4_quantization_label_rejects_legacy_v1_file() {
        let dir = tempdir("q4-label-v1");
        write_fake_q4_file(&dir.join("sample.q4"), 1, &[32], 32, 1);
        let err = detect_q4_quantization_label(&dir).unwrap_err();
        assert!(err.contains("legacy"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_uses_index_manifest_and_ignores_merged_cache_files() {
        let dir = tempdir("q4-merge-safe");
        write_fake_q4_file(&dir.join("layer0_qkv.q4"), 2, &[32], 32, 1);
        write_fake_q4_file(&dir.join("layer0_z.q4"), 2, &[32], 32, 1);
        // A runtime-created merge cache duplicating the two tensors
        // above -- must NOT be double-counted *from its own file* (it
        // has no manifest entry), but its bytes ARE real resident
        // duplicates of in_proj_qkv/in_proj_z, so each of those two
        // entries counts twice (see `q4_resident_bytes`).
        write_fake_q4_file(&dir.join("merged_qkvz_0_100_50.q4"), 2, &[64], 64, 2);

        let qkv_len = fs::metadata(dir.join("layer0_qkv.q4")).unwrap().len();
        let z_len = fs::metadata(dir.join("layer0_z.q4")).unwrap().len();

        let index = serde_json::json!([
            {"name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "file": "layer0_qkv.q4", "quantized": true, "shape": [32], "numel": 32},
            {"name": "model.language_model.layers.0.linear_attn.in_proj_z.weight", "file": "layer0_z.q4", "quantized": true, "shape": [32], "numel": 32},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert_eq!(inv.total_bytes, 2 * (qkv_len + z_len));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_resident_bytes_exceed_raw_on_disk_sum_for_expanded_tensors() {
        // What this test guards: several Q4/Metal tensor
        // categories are dequantized or duplicated at load time, so a
        // flat `meta.len()` sum under-reports real resident bytes. This
        // must fail if `inspect_q4_dir` reverts to summing raw on-disk
        // sizes unmodified.
        let dir = tempdir("q4-resident-expansion");
        // embed_tokens: expect 4.2x (dequantized f16 copy + separate
        // mmap for the logits GEMV).
        write_fake_q4_file(
            &dir.join("embed.q4"),
            2,
            &[32],
            32,
            10, // 10 blocks * 20 bytes = 200 on-disk bytes
        );
        // in_proj_a: expect 3.2x (dequantized to an f16 Metal buffer).
        write_fake_q4_file(&dir.join("proj_a.q4"), 2, &[32], 32, 10);
        // A plain mmap'd tensor (q_proj): expect exactly 1x, unchanged.
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 10);

        let embed_len = fs::metadata(dir.join("embed.q4")).unwrap().len();
        let proj_a_len = fs::metadata(dir.join("proj_a.q4")).unwrap().len();
        let q_proj_len = fs::metadata(dir.join("q_proj.q4")).unwrap().len();

        let index = serde_json::json!([
            {"name": "model.language_model.embed_tokens.weight", "file": "embed.q4", "quantized": true, "shape": [32], "numel": 32},
            {"name": "model.language_model.layers.0.linear_attn.in_proj_a.weight", "file": "proj_a.q4", "quantized": true, "shape": [32], "numel": 32},
            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();

        let naive_sum = embed_len + proj_a_len + q_proj_len;
        let expected = (embed_len as f64 * 4.2).round() as u64
            + (proj_a_len as f64 * 3.2).round() as u64
            + q_proj_len;
        assert!(
            inv.total_bytes > naive_sum,
            "resident estimate ({}) must exceed the naive on-disk sum ({naive_sum}) \
                 once embed_tokens/in_proj_a expansion is accounted for",
            inv.total_bytes
        );
        assert_eq!(inv.total_bytes, expected);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn q4_resident_bytes_classifies_every_known_category() {
        // Direct unit coverage for the classifier itself, independent of
        // the manifest/fallback-scan plumbing above -- also exercises
        // the sanitized-filename form (dots -> `_`, `.q4`/`.f16` suffix)
        // used by the no-manifest fallback path. `tie_word_embeddings`
        // is fixed at `true` throughout this test except where noted --
        // see `q4_resident_bytes_embed_tokens_scales_by_tie_word_embeddings`
        // for the tied/untied comparison this parameter exists for.
        assert_eq!(
            q4_resident_bytes("model.language_model.norm.weight", 100, true),
            200
        );
        assert_eq!(
            q4_resident_bytes("model_language_model_norm_weight.f16", 100, true),
            200
        );
        assert_eq!(
            q4_resident_bytes("model.language_model.layers.0.linear_attn.A_log", 100, true),
            200
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.linear_attn.dt_bias",
                100,
                true
            ),
            200
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                100,
                true
            ),
            200
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                100,
                true
            ),
            320
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                100,
                true
            ),
            320
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                100,
                true
            ),
            200
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                100,
                true
            ),
            200
        );
        assert_eq!(
            q4_resident_bytes("model.language_model.embed_tokens.weight", 100, true),
            420
        );
        assert_eq!(
            q4_resident_bytes("mtp.pre_fc_norm_embedding.weight", 100, true),
            200
        );
        assert_eq!(
            q4_resident_bytes("mtp.pre_fc_norm_hidden.weight", 100, true),
            200
        );
        assert_eq!(
            q4_resident_bytes("mtp_pre_fc_norm_embedding_weight.f16", 100, true),
            200
        );
        assert_eq!(
            q4_resident_bytes("mtp_pre_fc_norm_hidden_weight.f16", 100, true),
            200
        );
        // Unaffected categories stay at exactly 1x, regardless of
        // tie_word_embeddings (only embed_tokens reads that flag).
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                100,
                true
            ),
            100
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                100,
                false
            ),
            100
        );
        assert_eq!(
            q4_resident_bytes(
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                100,
                true
            ),
            100
        );
        assert_eq!(q4_resident_bytes("lm_head.weight", 100, false), 100);
    }

    #[test]
    fn q4_resident_bytes_embed_tokens_scales_by_tie_word_embeddings() {
        // #881: a tied checkpoint's `embed_tokens_q8`
        // binding keeps the raw Q4 mmap of `embed_tokens.weight` itself
        // (1x) on top of the dequantized f16 lookup buffer (3.2x) =
        // 4.2x. An untied checkpoint's `from_q4_dir` immediately
        // shadows that same binding with a fresh mmap of the separate
        // `lm_head.weight.q4` file, dropping the embedding's own Q4
        // mmap before the function returns -- only the 3.2x f16 dequant
        // survives, and `lm_head.weight` is counted on its own (as a
        // plain 1x, via a different `q4_resident_bytes` call for that
        // name, which does not match any expansion category).
        let tensor = "model.language_model.embed_tokens.weight";
        assert_eq!(q4_resident_bytes(tensor, 1000, true), 4200);
        assert_eq!(q4_resident_bytes(tensor, 1000, false), 3200);
        // lm_head itself is always a plain 1x mmap in the inventory,
        // independent of tie_word_embeddings.
        assert_eq!(q4_resident_bytes("lm_head.weight", 1000, true), 1000);
        assert_eq!(q4_resident_bytes("lm_head.weight", 1000, false), 1000);
    }

    #[test]
    fn inspect_q4_dir_manifest_flags_missing_file() {
        let dir = tempdir("q4-missing-file");
        let index = serde_json::json!([
            {"name": "some.tensor.weight", "file": "does_not_exist.q4", "quantized": true, "shape": [32], "numel": 32},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        write_fake_q4_file(&dir.join("sample.q4"), 2, &[32], 32, 1);

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert!(
            inv.missing_tensors
                .iter()
                .any(|m| m.contains("some.tensor.weight"))
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_falls_back_to_directory_scan_without_manifest() {
        let dir = tempdir("q4-no-manifest");
        write_fake_q4_file(&dir.join("a.q4"), 2, &[32], 32, 1);
        write_fake_q4_file(&dir.join("b.q4"), 2, &[32], 32, 1);
        write_fake_q4_file(&dir.join("merged_qkvz_x.q4"), 2, &[64], 64, 2);
        let a_len = fs::metadata(dir.join("a.q4")).unwrap().len();
        let b_len = fs::metadata(dir.join("b.q4")).unwrap().len();

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert_eq!(inv.total_bytes, a_len + b_len);
        assert_eq!(inv.tensor_count, 2);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_fallback_scan_applies_resident_bytes_by_sanitized_filename() {
        // The no-manifest fallback path only sees sanitized filenames
        // (q4_tensor_path: dots -> `_`), not the original dotted tensor
        // name -- this proves the classifier's suffix matching still
        // works against that form for an expanded category.
        let dir = tempdir("q4-no-manifest-embed");
        write_fake_q4_file(
            &dir.join("model_language_model_embed_tokens_weight.q4"),
            2,
            &[32],
            32,
            10,
        );
        write_fake_q4_file(
            &dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"),
            2,
            &[32],
            32,
            10,
        );
        let embed_len = fs::metadata(dir.join("model_language_model_embed_tokens_weight.q4"))
            .unwrap()
            .len();
        let q_proj_len =
            fs::metadata(dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"))
                .unwrap()
                .len();

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        let expected = (embed_len as f64 * 4.2).round() as u64 + q_proj_len;
        assert_eq!(inv.total_bytes, expected);
        assert!(inv.total_bytes > embed_len + q_proj_len);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_detects_mtp_tensors_via_manifest() {
        let dir = tempdir("q4-mtp-manifest");
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 1);
        write_fake_q4_file(&dir.join("mtp_norm.f16"), 2, &[32], 32, 1);
        let index = serde_json::json!([
            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
            {"name": "mtp.norm.weight", "file": "mtp_norm.f16", "quantized": false, "shape": [32], "numel": 32},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert!(inv.has_mtp_tensors);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_no_mtp_tensors_via_manifest_when_absent() {
        let dir = tempdir("q4-no-mtp-manifest");
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 1);
        let index = serde_json::json!([
            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert!(!inv.has_mtp_tensors);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_accepts_both_bare_array_and_quarot_object_manifest_shapes() {
        // Regression test for #626: `quantize_q4` writes
        // `quantize_index.json` as a bare array, but `quantize_quarot`
        // (ADR-051) writes an object -- `{"quarot_seed": ..., "tensors":
        // [...]}` -- so a loader can recover the QuaRot rotation seed
        // without parsing `config.json`. `doctor` crashed on the latter
        // shape with "invalid type: map, expected a sequence" before this
        // fix; both shapes must now load and produce equivalent
        // inventories for an identical tensor list.
        let bare_dir = tempdir("q4-manifest-bare-array");
        let object_dir = tempdir("q4-manifest-quarot-object");
        for dir in [&bare_dir, &object_dir] {
            write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 10);
        }
        let q_proj_len = fs::metadata(bare_dir.join("q_proj.q4")).unwrap().len();

        let tensors = serde_json::json!([
            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
        ]);
        fs::write(
            bare_dir.join("quantize_index.json"),
            serde_json::to_vec(&tensors).unwrap(),
        )
        .unwrap();

        let quarot_seed: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let object_manifest = serde_json::json!({
            "quarot_seed": quarot_seed,
            "tensors": [
                {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
            ],
        });
        fs::write(
            object_dir.join("quantize_index.json"),
            serde_json::to_vec(&object_manifest).unwrap(),
        )
        .unwrap();

        let cfg = Qwen35Config::qwen35_0_8b();
        let bare_inv = inspect_q4_dir(&bare_dir, &cfg)
            .expect("bare-array quantize_index.json (quantize_q4's shape) must load");
        let object_inv = inspect_q4_dir(&object_dir, &cfg)
            .expect("object-form quantize_index.json (quantize_quarot's shape) must load");

        assert_eq!(object_inv.total_bytes, bare_inv.total_bytes);
        assert_eq!(object_inv.total_bytes, q_proj_len);
        assert_eq!(object_inv.tensor_count, bare_inv.tensor_count);
        assert_eq!(object_inv.tensor_count, 1);
        assert_eq!(object_inv.missing_tensors, bare_inv.missing_tensors);
        assert_eq!(object_inv.has_mtp_tensors, bare_inv.has_mtp_tensors);

        fs::remove_dir_all(&bare_dir).ok();
        fs::remove_dir_all(&object_dir).ok();
    }

    // Manifest shape/malformation parsing itself (missing-field errors,
    // non-array `tensors`, scalar roots, oversized/truncated files) is
    // covered directly against `lattice_inference::quant::q4_manifest`
    // (issue #655 centralization). The tests here instead exercise
    // `inspect_q4_dir`'s use of that shared parser end-to-end, so a
    // malformed manifest surfaces as a `doctor`-level failure too.
    #[test]
    fn inspect_q4_dir_malformed_manifest_surfaces_precise_schema_error() {
        let dir = tempdir("q4-manifest-malformed");
        fs::write(dir.join("quantize_index.json"), br#"[{"name": "x"}]"#).unwrap();
        let cfg = Qwen35Config::qwen35_0_8b();
        let err = inspect_q4_dir(&dir, &cfg)
            .err()
            .expect("bare-array entry missing `file` must fail");
        assert!(
            err.contains("file"),
            "error must name the missing `file` field; got: {err}"
        );
        assert!(
            !err.contains("did not match any variant"),
            "error must not be the generic untagged-enum fallthrough; got: {err}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_detects_mtp_tensors_via_fallback_scan() {
        let dir = tempdir("q4-mtp-fallback");
        write_fake_q4_file(
            &dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"),
            2,
            &[32],
            32,
            1,
        );
        write_fake_q4_file(
            &dir.join("mtp_pre_fc_norm_embedding_weight.f16"),
            2,
            &[32],
            32,
            1,
        );

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert!(inv.has_mtp_tensors);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inspect_q4_dir_no_mtp_tensors_via_fallback_scan_when_absent() {
        let dir = tempdir("q4-no-mtp-fallback");
        write_fake_q4_file(
            &dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"),
            2,
            &[32],
            32,
            1,
        );

        let cfg = Qwen35Config::qwen35_0_8b();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert!(!inv.has_mtp_tensors);
        fs::remove_dir_all(&dir).ok();
    }

    // ---- system memory detection --------------------------------------

    #[test]
    fn detect_total_memory_bytes_is_plausible_when_known() {
        if let Some(bytes) = detect_total_memory_bytes() {
            assert!(bytes > 0);
            assert!(bytes < (1u64 << 50)); // sanity upper bound: 1 PiB
        }
    }

    // ---- build_report end-to-end --------------------------------------

    fn write_config_json(dir: &Path, cfg: &Qwen35Config) {
        let config_json = serde_json::json!({
            "text_config": {
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "vocab_size": cfg.vocab_size,
                "intermediate_size": cfg.intermediate_size,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "rope_theta": cfg.rope_theta,
                "partial_rotary_factor": cfg.partial_rotary_factor,
                "linear_num_key_heads": cfg.linear_num_key_heads,
                "linear_num_value_heads": cfg.linear_num_value_heads,
                "linear_key_head_dim": cfg.linear_key_head_dim,
                "linear_value_head_dim": cfg.linear_value_head_dim,
                "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "max_position_embeddings": cfg.max_position_embeddings,
                "eos_token_id": cfg.eos_token_id,
                "full_attention_interval": cfg.full_attention_interval,
            }
        });
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&config_json).unwrap(),
        )
        .unwrap();
    }

    fn write_complete_safetensors_fixture(dir: &Path, cfg: &Qwen35Config) {
        let tensors = required_tensor_fixture(cfg);
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);
        write_config_json(dir, cfg);
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    }

    #[test]
    fn build_report_happy_path_is_ready() {
        let dir = tempdir("report-happy");
        let cfg = Qwen35Config::qwen35_0_8b();
        write_complete_safetensors_fixture(&dir, &cfg);

        let report = build_report(&dir, None, Some(4096), Some(1u64 << 40)).unwrap();
        assert!(
            report.is_ready(),
            "blocking reasons: {:?}",
            report.blocking_reasons
        );
        assert_eq!(report.placement, Placement::Cpu);
        assert_eq!(report.requested_fits, Some(true));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_infeasible_context_via_memory_override_exits_not_ready() {
        let dir = tempdir("report-infeasible");
        let cfg = Qwen35Config::qwen35_0_8b();
        write_complete_safetensors_fixture(&dir, &cfg);

        // Force a tiny "machine": far less memory than the weights
        // alone need, so no context length fits.
        let report = build_report(&dir, None, Some(1), Some(1024)).unwrap();
        assert!(!report.is_ready());
        assert_eq!(report.requested_fits, Some(false));
        assert!(
            report
                .blocking_reasons
                .iter()
                .any(|r| r.contains("does not fit"))
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_context_exceeding_any_real_machine_is_not_ready() {
        // Belt-and-suspenders alternative to the override above: an
        // absurd requested context against REAL detected memory (no
        // override) must still be infeasible on any real machine.
        let dir = tempdir("report-absurd-context");
        let cfg = Qwen35Config::qwen35_0_8b();
        write_complete_safetensors_fixture(&dir, &cfg);

        let report = build_report(&dir, None, Some(usize::MAX / 2), None).unwrap();
        if report.available_memory_bytes.is_some() {
            assert_eq!(report.requested_fits, Some(false));
            assert!(!report.is_ready());
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_weights_exceed_ram_fails_without_explicit_context() {
        // Issue #875: doctor's own numbers said weights (64.60 GiB)
        // exceeded system memory (32.00 GiB) yet the verdict was still
        // "OK" because no `--context` was passed. Reproduce with no
        // requested context at all -- the fail-closed check must not
        // depend on `requested_fits`.
        let dir = tempdir("report-weights-exceed-ram-no-ctx");
        let cfg = Qwen35Config::qwen35_0_8b();
        write_complete_safetensors_fixture(&dir, &cfg);
        let weight_bytes = required_tensor_fixture(&cfg).len() as u64 * 64;

        let report = build_report(&dir, None, None, Some(weight_bytes - 1)).unwrap();
        assert!(!report.is_ready());
        assert_eq!(report.requested_context, None);
        assert!(
            report
                .blocking_reasons
                .iter()
                .any(|r| r.contains("exceeds detected system memory")),
            "blocking reasons: {:?}",
            report.blocking_reasons
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_zero_feasible_context_fails_without_explicit_context() {
        // Weights alone fit, but leave less than one KV-cache token's
        // worth of headroom -- max feasible context is 0. Must fail
        // closed even though no `--context` was requested.
        let dir = tempdir("report-zero-ctx-no-request");
        let cfg = Qwen35Config::qwen35_0_8b();
        write_complete_safetensors_fixture(&dir, &cfg);
        let weight_bytes = required_tensor_fixture(&cfg).len() as u64 * 64;
        let kv_bytes_per_token = cfg.kv_bytes_per_token(kv_cache_dtype_bytes()) as u64;
        assert!(
            kv_bytes_per_token > 1,
            "fixture assumption: at least 2 bytes/token of KV cache"
        );

        let report = build_report(&dir, None, None, Some(weight_bytes + 1)).unwrap();
        assert_eq!(report.max_context_len, Some(0));
        assert!(!report.is_ready());
        assert_eq!(report.requested_context, None);
        assert!(
            report
                .blocking_reasons
                .iter()
                .any(|r| r.contains("max feasible context length is 0 tokens")),
            "blocking reasons: {:?}",
            report.blocking_reasons
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_fits_within_ram_is_ready_without_explicit_context() {
        // Existing passing configurations must still report OK when no
        // `--context` is requested (acceptance criterion: no
        // regression on the happy path).
        let dir = tempdir("report-fits-no-ctx");
        let cfg = Qwen35Config::qwen35_0_8b();
        write_complete_safetensors_fixture(&dir, &cfg);

        let report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
        assert!(
            report.is_ready(),
            "blocking reasons: {:?}",
            report.blocking_reasons
        );
        assert!(report.max_context_len.unwrap_or(0) > 0);
        fs::remove_dir_all(&dir).ok();
    }

    /// Minimal Q4 fixture for the threshold-regression tests below: just
    /// two manifest entries (an `embed_tokens` tensor and one
    /// never-expanded tensor, e.g. a plain `q_proj`) rather than the
    /// full `qwen_required_tensor_names` set. `inspect_q4_dir` sums
    /// `total_bytes` only over manifest entries actually present, so
    /// this keeps the expected total small enough to state as a closed
    /// formula in the test -- computed independently of
    /// `q4_resident_bytes`/`inspect_q4_dir` themselves, so a regression
    /// in either is caught rather than silently absorbed into a
    /// self-referential expectation. `missing_tensors` will list the
    /// rest of `qwen_required_tensor_names(cfg)` as absent, which is
    /// irrelevant here: it drives a separate, unrelated blocking
    /// reason, not the "exceeds detected system memory" one under test.
    fn write_minimal_q4_fixture(dir: &Path, cfg: &Qwen35Config) -> (u64, u64) {
        write_fake_q4_file(&dir.join("embed.q4"), 2, &[32], 32, 5);
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 3);
        let embed_bytes = fs::metadata(dir.join("embed.q4")).unwrap().len();
        let q_proj_bytes = fs::metadata(dir.join("q_proj.q4")).unwrap().len();
        let index = serde_json::json!([
            {"name": "model.language_model.embed_tokens.weight", "file": "embed.q4", "quantized": true, "shape": [32], "numel": 32},
            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        write_config_json(dir, cfg);
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        (embed_bytes, q_proj_bytes)
    }

    #[test]
    fn build_report_q4_untied_embed_tokens_threshold_no_longer_overcounts() {
        // #881: before this fix, `q4_resident_bytes`
        // charged `embed_tokens` at a flat 4.2x on-disk size regardless
        // of `tie_word_embeddings`, even though an UNTIED checkpoint's
        // loader drops the embedding's own Q4 mmap and only keeps the
        // 3.2x f16 dequant (see `q4_resident_bytes`'s doc comment). That
        // stale 4.2x overcounted an untied checkpoint by exactly one
        // embedding-file's worth of bytes -- enough to push a
        // near-boundary checkpoint across the new hard-fail threshold
        // for a machine it would actually fit on.
        //
        // Reproduce at the boundary: pick a memory override strictly
        // between an INDEPENDENTLY hand-computed corrected (3.2x) total
        // and what the stale flat-4.2x estimate would have been for the
        // same on-disk bytes -- both computed here with the `* 3.2`/
        // `* 4.2` arithmetic directly, not by calling
        // `q4_resident_bytes`/`inspect_q4_dir`, so a regression back to
        // the flat-4.2x behavior actually pushes `report`'s real
        // `inventory.total_bytes` back over this test's override and
        // is caught.
        let dir = tempdir("report-q4-untied-threshold");
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.tie_word_embeddings = false;
        let (embed_bytes, q_proj_bytes) = write_minimal_q4_fixture(&dir, &cfg);

        let corrected_total = (embed_bytes as f64 * 3.2).round() as u64 + q_proj_bytes; // untied: 3.2x + 1x
        let stale_would_have_been = (embed_bytes as f64 * 4.2).round() as u64 + q_proj_bytes; // old flat 4.2x + 1x
        assert!(stale_would_have_been > corrected_total);
        let threshold_override = corrected_total + (stale_would_have_been - corrected_total) / 2;
        assert!(threshold_override > corrected_total);
        assert!(threshold_override < stale_would_have_been);

        let report = build_report(&dir, None, None, Some(threshold_override)).unwrap();
        assert!(
            !report
                .blocking_reasons
                .iter()
                .any(|r| r.contains("exceeds detected system memory")),
            "corrected Q4 estimate must fit within a budget between the corrected and \
                 stale flat-4.2x estimate; blocking reasons: {:?}",
            report.blocking_reasons
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_q4_true_over_budget_still_fails() {
        // Belt-and-suspenders alongside the threshold test above: a Q4
        // checkpoint that genuinely exceeds even the corrected
        // (tie_word_embeddings-aware) estimate must still fail closed
        // -- the fix must not turn into a blanket pass for Q4. Same
        // independent hand-computed formula as the test above.
        let dir = tempdir("report-q4-true-over-budget");
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.tie_word_embeddings = false;
        let (embed_bytes, q_proj_bytes) = write_minimal_q4_fixture(&dir, &cfg);

        let corrected_total = (embed_bytes as f64 * 3.2).round() as u64 + q_proj_bytes;
        let report = build_report(&dir, None, None, Some(corrected_total - 1)).unwrap();
        assert!(
            report
                .blocking_reasons
                .iter()
                .any(|r| r.contains("exceeds detected system memory")),
            "a checkpoint exceeding even the corrected estimate must still fail closed; \
                 blocking reasons: {:?}",
            report.blocking_reasons
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_missing_tokenizer_is_not_ready() {
        let dir = tempdir("report-no-tokenizer");
        let cfg = Qwen35Config::qwen35_0_8b();
        let tensors = required_tensor_fixture(&cfg);
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);
        write_config_json(&dir, &cfg);
        // No tokenizer.json written.

        let report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
        assert!(!report.is_ready());
        assert!(!report.tokenizer_present);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_unknown_format_is_err() {
        let dir = tempdir("report-unknown");
        let err = build_report(&dir, None, None, None).unwrap_err();
        assert!(err.contains("not a recognized model directory"));
        fs::remove_dir_all(&dir).ok();
    }

    // ---- #923: missing config.json is a hard, uniform error -----------
    //
    // Before the fix, this Safetensors branch silently substituted
    // `Qwen35Config::qwen35_2b()` and the Q4 branch silently substituted
    // `Qwen35Config::qwen36_27b()` -- two different guessed presets, and
    // neither told the caller a config.json was even missing. Both
    // branches now go through the same `Qwen35Config::from_model_dir`
    // fail-closed helper as every other loader in this crate. Reverting
    // either branch's call site back to the old exists-then-preset
    // pattern makes its half of this test fail (it would return `Ok`
    // instead of an error naming `config.json`).

    #[test]
    fn build_report_safetensors_missing_config_json_is_hard_error() {
        let dir = tempdir("report-safetensors-no-config");
        let cfg = Qwen35Config::qwen35_0_8b();
        let tensors = required_tensor_fixture(&cfg);
        let refs: Vec<(&str, &str, u64, u64)> = tensors
            .iter()
            .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
            .collect();
        write_fake_safetensors(&dir.join("model.safetensors"), &refs);
        // Deliberately no config.json and no tokenizer.json.

        let err = build_report(&dir, None, None, Some(1u64 << 40)).unwrap_err();
        assert!(
            err.contains("config.json"),
            "error must name the missing config.json, not a guessed preset: {err}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_report_q4_missing_config_json_is_hard_error() {
        let dir = tempdir("report-q4-no-config");
        write_fake_q4_file(&dir.join("embed.q4"), 2, &[32], 32, 5);
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 3);
        let index = serde_json::json!([
            {"name": "model.language_model.embed_tokens.weight", "file": "embed.q4", "quantized": true, "shape": [32], "numel": 32},
            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        // Deliberately no config.json.

        let err = build_report(&dir, None, None, Some(1u64 << 40)).unwrap_err();
        assert!(
            err.contains("config.json"),
            "error must name the missing config.json, not a guessed preset: {err}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    // ---- DoctorReport Display: MTP disclosure --------------------------

    fn minimal_doctor_report(has_mtp_tensors: bool) -> DoctorReport {
        // Constructed directly rather than through `build_report` -- this
        // targets the `Display` impl's MTP-disclosure branch in
        // isolation, without needing a full valid Q4 checkpoint fixture
        // (manifest + every required tensor + tokenizer.json) that no
        // other test in this module builds either.
        DoctorReport {
            model_dir: PathBuf::from("/fake/model"),
            format: crate::backend::ModelFormat::Q4,
            placement: Placement::Metal,
            quantization: "Q4_0".to_string(),
            tensor_count: 1,
            weight_bytes: 100,
            kv_bytes_per_token: 100,
            max_position_embeddings: 4096,
            metal_runtime_cache_cap: Some(METAL_RUNTIME_MAX_CACHE_LEN),
            available_memory_bytes: Some(1u64 << 40),
            max_context_len: Some(4096),
            requested_context: None,
            requested_fits: None,
            tokenizer_path: PathBuf::from("/fake/model/tokenizer.json"),
            tokenizer_present: true,
            missing_tensors: Vec::new(),
            blocking_reasons: Vec::new(),
            has_mtp_tensors,
        }
    }

    #[test]
    fn doctor_report_display_includes_mtp_disclosure_when_mtp_tensors_present() {
        let report = minimal_doctor_report(true);
        let text = format!("{report}");
        assert!(
            text.contains("MTP"),
            "report must disclose the uncounted MTP K/V cache when MTP tensors are present:\n{text}"
        );
    }

    #[test]
    fn doctor_report_display_omits_mtp_disclosure_when_no_mtp_tensors() {
        let report = minimal_doctor_report(false);
        let text = format!("{report}");
        assert!(
            !text.contains("MTP"),
            "report must not mention MTP for a directory with no MTP tensors:\n{text}"
        );
    }
}
