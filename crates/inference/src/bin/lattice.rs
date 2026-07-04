//! `lattice` CLI — interactive chat, HTTP serve, and preflight subcommands.
//!
//! # Usage
//!
//! ```text
//! lattice chat --model /path/to/model [--max-tokens 256] [--temperature 0.7]
//! lattice serve --model /path/to/model [--host 127.0.0.1] [--port 8080] [--max-tokens 256]
//! lattice doctor --model /path/to/model [--context 4096]
//! ```

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "lattice", about = "Pure-Rust transformer inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive chat with a model
    Chat {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Maximum tokens to generate per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
    /// Start HTTP server with OpenAI-compatible API
    Serve {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Host address to bind (default: 127.0.0.1; use 0.0.0.0 for LAN)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Maximum tokens to generate per request (default when request omits max_tokens)
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Model identifier echoed in responses (defaults to the model path basename)
        #[arg(long)]
        model_id: Option<String>,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
    /// Preflight check: memory fit and artifact compatibility, without
    /// loading any model weights (config + tensor index inspection only).
    Doctor {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Context length to check feasibility for. When omitted, only the
        /// maximum feasible context length is reported.
        #[arg(long)]
        context: Option<usize>,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// backend: model-directory format detection + Q4/Metal loading
//
// `lattice chat`/`lattice serve` originally only understood a safetensors
// directory (`model.safetensors` or a sharded index). This module adds
// support for native Q4 quantized directories (per-tensor `.q4` files, the
// output of `quantize_q4`) by detecting the format up front and routing to
// the Metal GPU forward pass. Safetensors directories are completely
// unaffected: `detect_format` returns `Safetensors` for them exactly as
// before, and the safetensors load path is untouched.
// ---------------------------------------------------------------------------

mod backend {
    use std::path::Path;

    /// The on-disk format of a model directory, decided before any tensor I/O.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ModelFormat {
        /// `model.safetensors` or `model.safetensors.index.json` present.
        Safetensors,
        /// No safetensors file, but at least one `*.q4` tensor file present
        /// (the output of `quantize_q4`).
        Q4,
        /// Neither a safetensors file nor a `.q4` file was found.
        Unknown,
    }

    /// Detect whether `dir` is a safetensors model directory, a native Q4
    /// quantized directory, or neither.
    ///
    /// Mirrors the detection heuristic already shipped in `chat_metal.rs` and
    /// `lattice_serve.rs`: a directory is Q4 when it has no safetensors file
    /// and contains at least one file whose name ends in `.q4`.
    pub fn detect_format(dir: &Path) -> ModelFormat {
        if dir.join("model.safetensors").exists()
            || dir.join("model.safetensors.index.json").exists()
        {
            return ModelFormat::Safetensors;
        }
        let has_q4_file = std::fs::read_dir(dir)
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
        if has_q4_file {
            ModelFormat::Q4
        } else {
            ModelFormat::Unknown
        }
    }

    /// Error message shown when a Q4 directory is passed to a binary that was
    /// built without the `metal-gpu` feature. Q4 inference only runs on the
    /// Metal GPU forward pass; there is no CPU fallback for `.q4` tensors, so
    /// this is a hard, fail-closed error rather than a silent degrade.
    ///
    /// Only reachable from the `#[cfg(not(feature = "metal-gpu"))]` call sites
    /// in `run_chat` / `main`; a `metal-gpu` build never calls this (it loads
    /// Q4 directories instead), so it is legitimately unused in that
    /// configuration rather than by mistake.
    #[cfg_attr(feature = "metal-gpu", allow(dead_code))]
    pub fn metal_gpu_required_message(dir: &Path) -> String {
        format!(
            "model directory '{}' is a native Q4 quantized checkpoint, which requires \
             the Metal GPU forward pass. This binary was built without the `metal-gpu` \
             feature. Rebuild with `--features \"f16 metal-gpu\"` (macOS only), or point \
             --model at a safetensors directory instead.",
            dir.display()
        )
    }

    /// Error message for a directory that is neither safetensors nor Q4.
    pub fn unrecognized_format_message(dir: &Path) -> String {
        format!(
            "'{}' is not a recognized model directory: no model.safetensors, \
             model.safetensors.index.json, or *.q4 tensor files were found",
            dir.display()
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;

        fn tempdir(name: &str) -> std::path::PathBuf {
            let mut dir = std::env::temp_dir();
            dir.push(format!(
                "lattice-backend-test-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            fs::create_dir_all(&dir).expect("create tempdir");
            dir
        }

        #[test]
        fn detect_format_safetensors_file() {
            let dir = tempdir("safetensors-file");
            fs::write(dir.join("model.safetensors"), b"stub").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_safetensors_index_only() {
            let dir = tempdir("safetensors-index");
            fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_q4_dir() {
            let dir = tempdir("q4");
            fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
            fs::write(dir.join("config.json"), b"{}").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Q4);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_prefers_safetensors_over_q4_files() {
            // A directory that (unusually) has both a safetensors file and a
            // stray .q4 file must resolve as Safetensors — the safetensors
            // loader path is untouched and takes priority.
            let dir = tempdir("mixed");
            fs::write(dir.join("model.safetensors"), b"stub").unwrap();
            fs::write(dir.join("leftover.q4"), b"stub").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_empty_dir_is_unknown() {
            let dir = tempdir("empty");
            assert_eq!(detect_format(&dir), ModelFormat::Unknown);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_unrelated_files_is_unknown() {
            let dir = tempdir("unrelated");
            fs::write(dir.join("readme.txt"), b"hello").unwrap();
            fs::write(dir.join("config.json"), b"{}").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Unknown);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn metal_gpu_required_message_mentions_rebuild_flags() {
            let msg = metal_gpu_required_message(Path::new("/tmp/some-q4-dir"));
            assert!(msg.contains("metal-gpu"));
            assert!(msg.contains("--features"));
        }

        #[test]
        fn unrecognized_format_message_mentions_expected_files() {
            let msg = unrecognized_format_message(Path::new("/tmp/bogus"));
            assert!(msg.contains("model.safetensors"));
            assert!(msg.contains(".q4"));
        }

        // Fail-closed without metal-gpu: a Q4 directory must never silently
        // fall back to the CPU safetensors loader. This test only compiles
        // (and only means anything) when the binary is built WITHOUT the
        // metal-gpu feature — it asserts that the code path this binary
        // would take for a Q4 directory is the explicit error message above,
        // never `Qwen35Model::from_safetensors`.
        #[cfg(not(feature = "metal-gpu"))]
        #[test]
        fn q4_dir_without_metal_gpu_feature_fails_closed() {
            let dir = tempdir("q4-no-metal");
            fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Q4);
            // Scope: detection + message content only. This proves a Q4 dir
            // classifies as `ModelFormat::Q4` (so the `run_chat`/`main` match
            // arms take the fail-closed branch, never
            // `Qwen35Model::from_safetensors`) and that the error names the
            // rebuild flags. It does not drive `run_chat`/`main` themselves —
            // those exit the process, which a unit test cannot cross.
            let msg = metal_gpu_required_message(&dir);
            assert!(msg.contains("metal-gpu"));
            fs::remove_dir_all(&dir).ok();
        }
    }
}

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

mod doctor {
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
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
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
                let shard_path = dir.join(&shard_name);
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

    #[derive(serde::Deserialize)]
    struct Q4IndexEntry {
        name: String,
        file: String,
    }

    /// `quantize_index.json`'s on-disk shape differs by writer: `quantize_q4`
    /// serializes a bare `Vec<Q4IndexEntry>`, while `quantize_quarot` (ADR-051)
    /// serializes an object, `{"quarot_seed": ..., "tensors": [...]}`, so a
    /// loader can recover the QuaRot rotation seed without parsing
    /// `config.json`. `doctor` only inventories tensors -- the seed is
    /// irrelevant here -- so both shapes normalize to the same entry list via
    /// [`Q4IndexManifest::into_entries`]. Untagged so either JSON shape
    /// deserializes into this one type (see also `MessageContent` below for
    /// the same pattern applied to the OpenAI-compatible chat API).
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Q4IndexManifest {
        Bare(Vec<Q4IndexEntry>),
        Wrapped { tensors: Vec<Q4IndexEntry> },
    }

    impl Q4IndexManifest {
        fn into_entries(self) -> Vec<Q4IndexEntry> {
            match self {
                Q4IndexManifest::Bare(entries) => entries,
                Q4IndexManifest::Wrapped { tensors } => tensors,
            }
        }
    }

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
    ///   embedding lookup (3.2x, same ratio as `in_proj_a`/`b`) AND
    ///   separately mmap'd at its own on-disk size for the GPU logits GEMV
    ///   (`embed_tokens_q8`) — 4.2x total. (In the untied-embeddings case
    ///   the logits mmap actually targets a separate `lm_head.weight.q4`
    ///   file instead, which is already its own correctly-1x manifest
    ///   entry; treating `embed_tokens` as a flat 4.2x in both cases is a
    ///   deliberate, harmless over-estimate rather than added branching on
    ///   `tie_word_embeddings` for a difference this small.)
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
            return (on_disk_bytes as f64 * 4.2).round() as u64;
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
        let file = std::fs::File::open(sample.path())
            .map_err(|e| format!("failed to open {}: {e}", sample.path().display()))?;
        lattice_inference::weights::q4_weights::read_q4_header(&file)
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

        let index_path = dir.join("quantize_index.json");
        if index_path.exists() {
            let bytes = std::fs::read(&index_path)
                .map_err(|e| format!("failed to read {}: {e}", index_path.display()))?;
            let entries: Vec<Q4IndexEntry> = serde_json::from_slice::<Q4IndexManifest>(&bytes)
                .map_err(|e| format!("{} is not valid JSON: {e}", index_path.display()))?
                .into_entries();

            let mut total_bytes = 0u64;
            let mut missing_tensors = Vec::new();
            let mut present_names: BTreeSet<String> = BTreeSet::new();
            let mut has_mtp_tensors = false;
            for entry in &entries {
                let file_path = dir.join(&entry.file);
                match std::fs::metadata(&file_path) {
                    Ok(meta) => {
                        total_bytes += q4_resident_bytes(&entry.name, meta.len());
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
                let config_path = model_dir.join("config.json");
                let cfg = if config_path.exists() {
                    Qwen35Config::from_config_json(&config_path)
                        .map_err(|e| format!("config.json parse failed: {e}"))?
                } else {
                    // Mirrors `Qwen35Model::from_safetensors`'s own fallback.
                    Qwen35Config::qwen35_2b()
                };
                let inventory = inspect_safetensors_dir(model_dir, &cfg)?;
                (Placement::Cpu, cfg, inventory)
            }
            crate::backend::ModelFormat::Q4 => {
                let config_path = model_dir.join("config.json");
                let cfg = if config_path.exists() {
                    Qwen35Config::from_config_json(&config_path)
                        .map_err(|e| format!("config.json parse failed: {e}"))?
                } else {
                    // Mirrors `load_q4_config`'s own fallback (that function
                    // is `metal-gpu`-only; `doctor` must work without the
                    // feature too, so the two-line fallback is duplicated
                    // here rather than shared across the cfg-gate).
                    Qwen35Config::qwen36_27b()
                };
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
            // The Major finding this test guards: several Q4/Metal tensor
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
            // used by the no-manifest fallback path.
            assert_eq!(
                q4_resident_bytes("model.language_model.norm.weight", 100),
                200
            );
            assert_eq!(
                q4_resident_bytes("model_language_model_norm_weight.f16", 100),
                200
            );
            assert_eq!(
                q4_resident_bytes("model.language_model.layers.0.linear_attn.A_log", 100),
                200
            );
            assert_eq!(
                q4_resident_bytes("model.language_model.layers.0.linear_attn.dt_bias", 100),
                200
            );
            assert_eq!(
                q4_resident_bytes(
                    "model.language_model.layers.0.linear_attn.conv1d.weight",
                    100
                ),
                200
            );
            assert_eq!(
                q4_resident_bytes(
                    "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                    100
                ),
                320
            );
            assert_eq!(
                q4_resident_bytes(
                    "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                    100
                ),
                320
            );
            assert_eq!(
                q4_resident_bytes(
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                    100
                ),
                200
            );
            assert_eq!(
                q4_resident_bytes(
                    "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                    100
                ),
                200
            );
            assert_eq!(
                q4_resident_bytes("model.language_model.embed_tokens.weight", 100),
                420
            );
            assert_eq!(
                q4_resident_bytes("mtp.pre_fc_norm_embedding.weight", 100),
                200
            );
            assert_eq!(q4_resident_bytes("mtp.pre_fc_norm_hidden.weight", 100), 200);
            assert_eq!(
                q4_resident_bytes("mtp_pre_fc_norm_embedding_weight.f16", 100),
                200
            );
            assert_eq!(
                q4_resident_bytes("mtp_pre_fc_norm_hidden_weight.f16", 100),
                200
            );
            // Unaffected categories stay at exactly 1x.
            assert_eq!(
                q4_resident_bytes("model.language_model.layers.0.self_attn.q_proj.weight", 100),
                100
            );
            assert_eq!(
                q4_resident_bytes("model.language_model.layers.0.mlp.gate_proj.weight", 100),
                100
            );
            assert_eq!(
                q4_resident_bytes(
                    "model.language_model.layers.0.linear_attn.out_proj.weight",
                    100
                ),
                100
            );
            assert_eq!(q4_resident_bytes("lm_head.weight", 100), 100);
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
}

// ---------------------------------------------------------------------------
// chat subcommand
// ---------------------------------------------------------------------------

/// Load `config.json` for a Q4 directory, falling back to the Qwen3.6-27B
/// default config (matching `chat_metal.rs` / `lattice_serve.rs`) when the
/// directory has none, with a visible warning so a missing config.json is
/// never silently misinterpreted as intentional.
#[cfg(feature = "metal-gpu")]
fn load_q4_config(
    dir: &std::path::Path,
) -> Result<lattice_inference::model::qwen35_config::Qwen35Config, String> {
    let config_path = dir.join("config.json");
    if config_path.exists() {
        lattice_inference::model::qwen35_config::Qwen35Config::from_config_json(&config_path)
            .map_err(|e| format!("config.json parse failed: {e}"))
    } else {
        eprintln!(
            "Warning: {} has no config.json; falling back to the Qwen3.6-27B default config.",
            dir.display()
        );
        Ok(lattice_inference::model::qwen35_config::Qwen35Config::qwen36_27b())
    }
}

/// Metal-GPU chat backend: owns a `MetalQwen35State` plus the tokenizer and
/// context-window cap needed to serve `generate`/`generate_streaming` calls
/// the same way the CPU (`Qwen35Model`) backend does.
///
/// `MetalQwen35State` is `!Send` (it owns raw `metal::*` FFI objects), so
/// this type must never be shared across threads. `run_chat`'s REPL uses it
/// directly on the calling thread; the `serve` module never constructs one
/// on an async task — it lives on a dedicated worker thread instead (see
/// `serve::spawn_metal_worker`).
#[cfg(feature = "metal-gpu")]
struct MetalChatBackend {
    state: lattice_inference::forward::metal_qwen35::MetalQwen35State,
    tokenizer: lattice_inference::tokenizer::bpe::BpeTokenizer,
}

#[cfg(feature = "metal-gpu")]
impl MetalChatBackend {
    /// `max_cache_len` bounds the KV cache (and therefore the usable context
    /// window). 4096 matches the cap used by `chat_metal.rs`.
    const MAX_CACHE_LEN: usize = 4096;

    /// `tokenizer_dir` overrides where `tokenizer.json` is read from, for Q4
    /// directories that were produced without a co-located tokenizer. `None`
    /// resolves it from `dir` itself (the common case: Q4 dirs ship it).
    fn load(
        dir: &std::path::Path,
        tokenizer_dir: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let tokenizer_path = tokenizer_dir.unwrap_or(dir).join("tokenizer.json");
        let tokenizer =
            lattice_inference::tokenizer::bpe::BpeTokenizer::from_tokenizer_json(&tokenizer_path)
                .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?;
        let cfg = load_q4_config(dir)?;
        let state = lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir(
            dir,
            &tokenizer_path,
            &cfg,
            Self::MAX_CACHE_LEN,
        )
        .map_err(|e| format!("Q4 model load failed: {e}"))?;
        Ok(Self { state, tokenizer })
    }

    fn generate(
        &mut self,
        prompt: &str,
        gen_cfg: &lattice_inference::model::qwen35_config::GenerateConfig,
    ) -> lattice_inference::model::qwen35_config::GenerateOutput {
        self.state.generate(prompt, &self.tokenizer, gen_cfg)
    }
}

fn run_chat(model_path: &str, max_tokens: usize, temperature: f32, tokenizer_dir: Option<&str>) {
    use std::io::{BufRead, Write};
    use std::path::Path;

    let path = Path::new(model_path);
    let format = backend::detect_format(path);
    #[cfg(feature = "metal-gpu")]
    let tokenizer_dir_path = tokenizer_dir.map(Path::new);
    #[cfg(not(feature = "metal-gpu"))]
    let _ = tokenizer_dir;

    eprintln!("Loading model from {model_path}...");

    enum Backend {
        Cpu(Box<lattice_inference::model::qwen35::Qwen35Model>),
        #[cfg(feature = "metal-gpu")]
        Metal(Box<MetalChatBackend>),
    }

    let mut model = match format {
        backend::ModelFormat::Safetensors => {
            match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(path) {
                Ok(m) => Backend::Cpu(Box::new(m)),
                Err(e) => {
                    eprintln!("Error: failed to load model: {e}");
                    std::process::exit(1);
                }
            }
        }
        backend::ModelFormat::Q4 => {
            #[cfg(feature = "metal-gpu")]
            {
                match MetalChatBackend::load(path, tokenizer_dir_path) {
                    Ok(m) => Backend::Metal(Box::new(m)),
                    Err(e) => {
                        eprintln!("Error: failed to load Q4 model: {e}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(feature = "metal-gpu"))]
            {
                eprintln!("Error: {}", backend::metal_gpu_required_message(path));
                std::process::exit(1);
            }
        }
        backend::ModelFormat::Unknown => {
            eprintln!("Error: {}", backend::unrecognized_format_message(path));
            std::process::exit(1);
        }
    };
    eprintln!("Model loaded. Type 'exit' or 'quit' to stop.\n");

    let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        temperature,
        ..Default::default()
    };

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let prompt = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        };
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            break;
        }

        match &mut model {
            Backend::Cpu(m) => match m.generate(trimmed, &gen_cfg) {
                Ok(output) => {
                    let _ = writeln!(stdout, "{}", output.text);
                    let _ = writeln!(
                        stdout,
                        "[{} prompt tokens, {} generated]",
                        output.prompt_tokens, output.generated_tokens
                    );
                }
                Err(e) => {
                    eprintln!("Generation error: {e}");
                }
            },
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(m) => {
                let output = m.generate(trimmed, &gen_cfg);
                let _ = writeln!(stdout, "{}", output.text);
                let _ = writeln!(
                    stdout,
                    "[{} prompt tokens, {} generated]",
                    output.prompt_tokens, output.generated_tokens
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// serve subcommand: OpenAI-compatible HTTP API
// ---------------------------------------------------------------------------

mod serve {
    use axum::{
        Json, Router,
        extract::{DefaultBodyLimit, State},
        http::StatusCode,
        response::{
            IntoResponse, Response,
            sse::{Event, KeepAlive, Sse},
        },
        routing::{get, post},
    };
    use futures::StreamExt as _;
    use lattice_inference::Tokenizer;
    #[cfg(feature = "metal-gpu")]
    use lattice_inference::model::qwen35_config::GenerateConfig;
    use lattice_inference::model::qwen35_config::GenerateOutput;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Request body cap: 1 MiB.  Requests above this return HTTP 413.
    const REQUEST_BODY_LIMIT_BYTES: usize = 1_048_576;

    // -----------------------------------------------------------------------
    // Model backend: CPU (safetensors) or Metal GPU (native Q4)
    // -----------------------------------------------------------------------

    /// One generation request handed to the Metal GPU worker thread.
    ///
    /// `MetalQwen35State` owns raw `metal::*` FFI objects and is `!Send`, so it
    /// cannot be moved into a `tokio::task::spawn_blocking` closure the way the
    /// CPU model is (`Arc<Qwen35Model>` is `Send + Sync`; `MetalQwen35State` is
    /// neither). Instead the Metal state lives on ONE dedicated OS thread for
    /// the whole process lifetime, and async handlers ship it a `MetalJob` over
    /// an unbounded `tokio::sync::mpsc` channel — the same design already
    /// shipped in `lattice_serve.rs`. `on_token` is called synchronously from
    /// the worker thread for each streamed delta; returning `false` stops
    /// generation early (client disconnected).
    ///
    /// This serializes ALL Metal generation onto one thread: two concurrent
    /// requests to a Q4-backed `lattice serve` run back-to-back, not in
    /// parallel. That is correct for a single-GPU local engine (the same
    /// default ollama uses) and is documented here rather than hidden behind
    /// an innocuous-looking channel send.
    #[cfg(feature = "metal-gpu")]
    struct MetalJob {
        prompt: String,
        gen_cfg: GenerateConfig,
        on_token: Box<dyn FnMut(&str) -> bool + Send>,
        reply: tokio::sync::oneshot::Sender<GenerateOutput>,
    }

    /// Handle to the Metal GPU worker thread. Cheaply `Clone` (an `mpsc`
    /// sender), `Send + Sync`, so it can live in `AppState` like the CPU
    /// `Arc<Qwen35Model>` does — only the underlying `MetalQwen35State` is
    /// confined to the worker thread.
    #[cfg(feature = "metal-gpu")]
    #[derive(Clone)]
    pub struct MetalHandle {
        jobs: tokio::sync::mpsc::UnboundedSender<MetalJob>,
    }

    #[cfg(feature = "metal-gpu")]
    impl MetalHandle {
        /// Load the Q4 model on a new dedicated worker thread and return a
        /// handle once loading succeeds. Loading happens synchronously (the
        /// caller blocks until the model is ready or loading fails) so that
        /// `lattice serve`'s startup sequence keeps its existing "load, then
        /// bind, then listen" ordering and fails closed before ever binding
        /// the socket.
        fn spawn(
            model_dir: std::path::PathBuf,
            tokenizer_path: std::path::PathBuf,
            tokenizer: Arc<lattice_inference::tokenizer::bpe::BpeTokenizer>,
        ) -> Result<Self, String> {
            let (job_tx, mut job_rx) = tokio::sync::mpsc::unbounded_channel::<MetalJob>();
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

            std::thread::spawn(move || {
                let cfg = match super::load_q4_config(&model_dir) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let mut state =
                    match lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir(
                        &model_dir,
                        &tokenizer_path,
                        &cfg,
                        super::MetalChatBackend::MAX_CACHE_LEN,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!("Q4 model load failed: {e}")));
                            return;
                        }
                    };
                let _ = ready_tx.send(Ok(()));

                while let Some(job) = job_rx.blocking_recv() {
                    let mut on_token = job.on_token;
                    let output = state.generate_streaming(
                        &job.prompt,
                        &tokenizer,
                        &job.gen_cfg,
                        |delta, _token_id| on_token(delta),
                    );
                    let _ = job.reply.send(output);
                }
            });

            match ready_rx.recv() {
                Ok(Ok(())) => Ok(Self { jobs: job_tx }),
                Ok(Err(e)) => Err(e),
                Err(_) => Err("Metal worker thread exited before loading finished".to_string()),
            }
        }

        /// Run one generation on the worker thread, forwarding each token
        /// delta to `on_token`. Returns the full `GenerateOutput` (including
        /// `stopped`/`stop_reason`) so callers can compute `finish_reason`
        /// with the exact same `finish_reason_for` helper the CPU path uses.
        async fn generate_streaming(
            &self,
            prompt: String,
            gen_cfg: GenerateConfig,
            on_token: impl FnMut(&str) -> bool + Send + 'static,
        ) -> Result<GenerateOutput, ApiError> {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            let job = MetalJob {
                prompt,
                gen_cfg,
                on_token: Box::new(on_token),
                reply: reply_tx,
            };
            self.jobs.send(job).map_err(|_| ApiError::Internal {
                message: "inference worker is not running".to_string(),
            })?;
            reply_rx.await.map_err(|_| ApiError::Internal {
                message: "inference worker dropped the request".to_string(),
            })
        }
    }

    /// The two ways `AppState` can run generation: the original CPU
    /// (safetensors) path via `Arc<Qwen35Model>`, or the Metal GPU (native
    /// Q4) path via a worker-thread handle. Both variants funnel into the
    /// same request handler code below — `chat_completions` branches on this
    /// enum in exactly two places (streaming and non-streaming) rather than
    /// duplicating the handler.
    #[derive(Clone)]
    pub enum ModelBackend {
        Cpu(Arc<lattice_inference::model::qwen35::Qwen35Model>),
        #[cfg(feature = "metal-gpu")]
        Metal {
            handle: MetalHandle,
            tokenizer: Arc<lattice_inference::tokenizer::bpe::BpeTokenizer>,
            max_context: usize,
        },
    }

    impl ModelBackend {
        pub fn tokenize_len(&self, text: &str) -> usize {
            match self {
                ModelBackend::Cpu(m) => m.tokenizer().tokenize(text).real_length,
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { tokenizer, .. } => tokenizer.tokenize(text).real_length,
            }
        }

        pub fn max_context(&self) -> usize {
            match self {
                ModelBackend::Cpu(m) => m.max_context(),
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { max_context, .. } => *max_context,
            }
        }

        /// Load a native Q4 checkpoint on a dedicated Metal worker thread and
        /// return the `ModelBackend::Metal` handle plus the resolved context
        /// window, for `main()`'s `Command::Serve` startup sequence.
        #[cfg(feature = "metal-gpu")]
        pub fn spawn_metal(
            model_dir: std::path::PathBuf,
            tokenizer_dir: Option<std::path::PathBuf>,
        ) -> Result<(Self, usize), String> {
            let tokenizer_path = tokenizer_dir
                .as_deref()
                .unwrap_or(&model_dir)
                .join("tokenizer.json");
            let tokenizer = Arc::new(
                lattice_inference::tokenizer::bpe::BpeTokenizer::from_tokenizer_json(
                    &tokenizer_path,
                )
                .map_err(|e| {
                    format!("tokenizer load failed ({}): {e}", tokenizer_path.display())
                })?,
            );
            let max_context = super::MetalChatBackend::MAX_CACHE_LEN;
            let handle = MetalHandle::spawn(model_dir, tokenizer_path, Arc::clone(&tokenizer))?;
            Ok((
                ModelBackend::Metal {
                    handle,
                    tokenizer,
                    max_context,
                },
                max_context,
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Shared application state
    // -----------------------------------------------------------------------

    /// State shared across all request handlers via axum's `State` extractor.
    #[derive(Clone)]
    pub struct AppState {
        /// The loaded model backend (CPU safetensors or Metal GPU Q4).
        pub model: ModelBackend,
        /// Default `max_tokens` value used when a request omits the field.
        /// Set from the `--max-tokens` CLI flag passed to `lattice serve`.
        pub default_max_tokens: usize,
        /// Hard upper bound on `max_tokens` accepted from any request.
        /// Prevents callers from requesting unbounded generation.
        pub max_tokens_cap: usize,
        /// Canonical model identifier echoed in every response.
        /// Derived from the `--model-id` flag or the model path basename.
        pub model_id: String,
        /// Monotonically increasing counter used to make response IDs unique
        /// across concurrent requests within the same second.
        pub request_counter: Arc<AtomicU64>,
    }

    // -----------------------------------------------------------------------
    // Error type
    // -----------------------------------------------------------------------

    /// Structured HTTP error that serialises to the OpenAI error envelope so
    /// that clients can parse failure responses uniformly.
    #[derive(Debug)]
    pub enum ApiError {
        /// Caller mistake — HTTP 400.
        BadRequest { message: String, code: &'static str },
        /// Request body exceeds size limit — HTTP 413.
        PayloadTooLarge { message: String },
        /// Server-side failure — HTTP 500.
        Internal { message: String },
    }

    #[derive(Serialize)]
    struct ErrorBody {
        error: ErrorDetail,
    }

    #[derive(Serialize)]
    struct ErrorDetail {
        message: String,
        r#type: &'static str,
        code: String,
        param: Option<String>,
    }

    impl IntoResponse for ApiError {
        fn into_response(self) -> Response {
            match self {
                ApiError::BadRequest { message, code } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "invalid_request_error",
                            code: code.to_string(),
                            param: None,
                        },
                    });
                    (StatusCode::BAD_REQUEST, body).into_response()
                }
                ApiError::PayloadTooLarge { message } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "invalid_request_error",
                            code: "request_body_too_large".to_string(),
                            param: None,
                        },
                    });
                    (StatusCode::PAYLOAD_TOO_LARGE, body).into_response()
                }
                ApiError::Internal { message } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "server_error",
                            code: "internal_error".to_string(),
                            param: None,
                        },
                    });
                    (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Request / response types
    // -----------------------------------------------------------------------

    /// OpenAI-compatible chat completions request.
    ///
    /// Known-but-unsupported fields (`stream=true`, `tools`, `tool_choice`,
    /// `logprobs=true`, `n > 1`, `response_format` other than `"text"`) are
    /// parsed and explicitly rejected with HTTP 400 rather than silently dropped.
    /// `stop` is accepted and parsed into string-level stop sequences.
    /// Unknown fields are ignored by default (serde default).
    #[derive(Deserialize)]
    pub struct ChatCompletionRequest {
        /// Required: must match the served model identifier.
        pub model: String,
        pub messages: Vec<Message>,
        /// Generation token budget.  Use at most one of `max_tokens` /
        /// `max_completion_tokens`; if both are present they must agree.
        pub max_tokens: Option<usize>,
        /// Alias for `max_tokens` (current OpenAI naming).
        pub max_completion_tokens: Option<usize>,
        pub temperature: Option<f32>,
        /// Nucleus sampling probability mass.  Mapped into `GenerateConfig`.
        pub top_p: Option<f32>,
        /// SSE streaming — not yet supported; rejected with 400.
        pub stream: Option<bool>,
        /// Stop sequences — a JSON string or array of strings (up to 4, non-empty).
        /// Parsed by `parse_stop_strings`; null/absent → empty vec (no stops).
        pub stop: Option<Value>,
        /// Deterministic sampling seed.  Mapped into `GenerateConfig`.
        pub seed: Option<u64>,
        /// Response format constraint — only `"text"` is accepted.
        pub response_format: Option<ResponseFormat>,
        /// Tool definitions — not supported; rejected with 400.
        pub tools: Option<Value>,
        /// Tool choice — not supported; rejected with 400.
        pub tool_choice: Option<Value>,
        /// Log-probabilities — not supported; rejected with 400.
        pub logprobs: Option<bool>,
        /// Number of completions — only `1` is accepted.
        pub n: Option<usize>,
    }

    #[derive(Deserialize)]
    pub struct ResponseFormat {
        pub r#type: String,
    }

    /// Message content: either a plain string or an array of content parts.
    /// Non-text parts (image, audio, file) are rejected with HTTP 400.
    #[derive(Deserialize)]
    #[serde(untagged)]
    pub enum MessageContent {
        Text(String),
        Parts(Vec<ContentPart>),
    }

    #[derive(Deserialize)]
    pub struct ContentPart {
        #[serde(rename = "type")]
        pub kind: String,
        pub text: Option<String>,
    }

    #[derive(Deserialize)]
    pub struct Message {
        pub role: String,
        pub content: MessageContent,
    }

    #[derive(Serialize)]
    pub struct ChatCompletionResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<Choice>,
        pub usage: Usage,
    }

    #[derive(Serialize)]
    pub struct Choice {
        pub index: usize,
        pub message: ResponseMessage,
        pub finish_reason: String,
    }

    #[derive(Serialize)]
    pub struct ResponseMessage {
        pub role: String,
        pub content: String,
    }

    #[derive(Serialize)]
    pub struct Usage {
        pub prompt_tokens: usize,
        pub completion_tokens: usize,
        pub total_tokens: usize,
    }

    #[derive(Serialize)]
    pub struct HealthResponse {
        pub status: &'static str,
    }

    // -----------------------------------------------------------------------
    // SSE streaming types
    // -----------------------------------------------------------------------

    /// Internal channel message type for the streaming generation path.
    ///
    /// `spawn_blocking` runs the sync `generate_streaming` call on a blocking
    /// thread and sends incremental deltas through an unbounded channel.  The
    /// async SSE handler reads from the other end and maps these messages to
    /// OpenAI `chat.completion.chunk` events.
    pub enum StreamMsg {
        /// One incremental text delta from the model.
        Delta(String),
        /// Generation finished normally; carries the OpenAI finish reason.
        Done { finish_reason: &'static str },
        /// Generation failed (invariant violation or engine error).
        Failed,
    }

    /// Top-level chunk object serialised into each `data:` SSE event.
    #[derive(Serialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub object: &'static str,
        pub created: u64,
        pub model: String,
        pub choices: Vec<ChunkChoice>,
    }

    #[derive(Serialize)]
    pub struct ChunkChoice {
        pub index: usize,
        pub delta: ChunkDelta,
        /// Null while streaming, set on the final choice.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub finish_reason: Option<&'static str>,
    }

    /// The `delta` field of a streaming chunk.
    ///
    /// Exactly one of `role` / `content` is set per chunk:
    /// - First chunk: `role = "assistant"`, no content.
    /// - Subsequent content chunks: `content = <text>`, no role.
    /// - Final finish chunk: both absent (empty delta `{}`).
    #[derive(Serialize)]
    pub struct ChunkDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub role: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
    }

    // -----------------------------------------------------------------------
    // Validation helpers — pure functions, no model required, easily tested
    // -----------------------------------------------------------------------

    /// Resolve the effective `max_tokens`, rejecting zero, values above the
    /// server cap, and conflicting `max_tokens` / `max_completion_tokens`.
    fn validate_max_tokens(
        req_max: Option<usize>,
        req_max_completion: Option<usize>,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<usize, ApiError> {
        let effective = match (req_max, req_max_completion) {
            (None, None) => default_max_tokens,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (Some(a), Some(b)) if a == b => a,
            (Some(a), Some(b)) => {
                return Err(ApiError::BadRequest {
                    message: format!(
                        "max_tokens ({a}) and max_completion_tokens ({b}) differ; supply only one"
                    ),
                    code: "invalid_request",
                });
            }
        };
        if effective == 0 {
            return Err(ApiError::BadRequest {
                message: "max_tokens must be at least 1".to_string(),
                code: "invalid_max_tokens",
            });
        }
        if effective > max_tokens_cap {
            return Err(ApiError::BadRequest {
                message: format!("max_tokens {effective} exceeds server limit {max_tokens_cap}"),
                code: "max_tokens_exceeds_limit",
            });
        }
        Ok(effective)
    }

    /// Validate `temperature` is in `[0.0, 2.0]`.
    fn validate_temperature(value: Option<f32>) -> Result<f32, ApiError> {
        let temperature = value.unwrap_or(0.7);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::BadRequest {
                message: "temperature must be between 0 and 2".to_string(),
                code: "invalid_temperature",
            });
        }
        Ok(temperature)
    }

    /// Validate `top_p` is in `(0.0, 1.0]`.
    fn validate_top_p(value: Option<f32>) -> Result<f32, ApiError> {
        let top_p = value.unwrap_or(0.9);
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(ApiError::BadRequest {
                message: "top_p must be greater than 0 and at most 1".to_string(),
                code: "invalid_top_p",
            });
        }
        Ok(top_p)
    }

    /// Parse the OpenAI `stop` field into a `Vec<String>`.
    ///
    /// Accepted forms:
    /// - `null` / absent → empty vec (no string-level stops)
    /// - a JSON string → `vec![s]`
    /// - a JSON array of 1–4 non-empty strings → that vec
    ///
    /// Returns `Err(BadRequest)` for:
    /// - an empty array
    /// - an array with more than 4 elements
    /// - any array element that is not a string
    /// - any stop string that is empty
    fn parse_stop_strings(stop: &Option<Value>) -> Result<Vec<String>, ApiError> {
        match stop {
            None => Ok(vec![]),
            Some(Value::Null) => Ok(vec![]),
            Some(Value::String(s)) => {
                if s.is_empty() {
                    return Err(ApiError::BadRequest {
                        message: "stop string must not be empty".to_string(),
                        code: "invalid_stop",
                    });
                }
                Ok(vec![s.clone()])
            }
            Some(Value::Array(arr)) => {
                if arr.is_empty() {
                    return Err(ApiError::BadRequest {
                        message: "stop array must not be empty".to_string(),
                        code: "invalid_stop",
                    });
                }
                if arr.len() > 4 {
                    return Err(ApiError::BadRequest {
                        message: format!("stop array has {} elements; maximum is 4", arr.len()),
                        code: "invalid_stop",
                    });
                }
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    match item {
                        Value::String(s) => {
                            if s.is_empty() {
                                return Err(ApiError::BadRequest {
                                    message: "stop string must not be empty".to_string(),
                                    code: "invalid_stop",
                                });
                            }
                            out.push(s.clone());
                        }
                        _ => {
                            return Err(ApiError::BadRequest {
                                message: "each element of stop must be a string".to_string(),
                                code: "invalid_stop",
                            });
                        }
                    }
                }
                Ok(out)
            }
            Some(_) => Err(ApiError::BadRequest {
                message: "stop must be a string or array of strings".to_string(),
                code: "invalid_stop",
            }),
        }
    }

    /// Reject OpenAI fields that are parsed but not yet implemented.
    ///
    /// Note: `stream=true` is now handled by the streaming path in `chat_completions`
    /// and is intentionally NOT rejected here.
    fn reject_unsupported(req: &ChatCompletionRequest) -> Result<(), ApiError> {
        if req.tools.is_some() || req.tool_choice.is_some() {
            return Err(ApiError::BadRequest {
                message: "tools and tool_choice are not supported by this server".to_string(),
                code: "unsupported_feature",
            });
        }
        if req.logprobs.unwrap_or(false) {
            return Err(ApiError::BadRequest {
                message: "logprobs is not supported by this server".to_string(),
                code: "unsupported_feature",
            });
        }
        if req.n.unwrap_or(1) > 1 {
            return Err(ApiError::BadRequest {
                message: "n > 1 is not supported".to_string(),
                code: "unsupported_feature",
            });
        }
        if let Some(fmt) = &req.response_format
            && fmt.r#type != "text"
        {
            return Err(ApiError::BadRequest {
                message: format!(
                    "response_format.type '{}' is not supported; use 'text'",
                    fmt.r#type
                ),
                code: "unsupported_feature",
            });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Extract a plain text string from a message content value.
    /// Returns `Err` for non-text content parts (image, audio, file).
    fn message_text(content: &MessageContent) -> Result<String, ApiError> {
        match content {
            MessageContent::Text(text) => Ok(text.clone()),
            MessageContent::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    if part.kind != "text" {
                        return Err(ApiError::BadRequest {
                            message: format!(
                                "content part type '{}' is not supported; only 'text' parts are accepted",
                                part.kind
                            ),
                            code: "unsupported_feature",
                        });
                    }
                    out.push_str(part.text.as_deref().unwrap_or(""));
                }
                Ok(out)
            }
        }
    }

    /// Build a single prompt string from the full message list using Qwen ChatML format.
    ///
    /// Format (one block per message, in order):
    /// ```text
    /// <|im_start|>system
    /// {content}<|im_end|>
    /// <|im_start|>user
    /// {content}<|im_end|>
    /// <|im_start|>assistant
    /// {content}<|im_end|>
    /// ```
    /// The final line is the open generation prompt `<|im_start|>assistant\n` — no closing
    /// `<|im_end|>` — so the model generates from there.
    ///
    /// Only the roles `system`, `user`, and `assistant` are supported. Any other role
    /// returns `Err` so the handler can respond with HTTP 400.
    fn render_prompt(messages: &[Message]) -> Result<String, ApiError> {
        let mut buf = String::new();
        for msg in messages {
            let content = message_text(&msg.content)?;
            match msg.role.as_str() {
                "system" | "user" | "assistant" => {
                    buf.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        msg.role, content
                    ));
                }
                "tool" | "developer" => {
                    return Err(ApiError::BadRequest {
                        message: format!("role '{}' is not supported by this server", msg.role),
                        code: "unsupported_feature",
                    });
                }
                other => {
                    return Err(ApiError::BadRequest {
                        message: format!(
                            "unsupported role '{other}'; must be 'system', 'user', or 'assistant'"
                        ),
                        code: "invalid_role",
                    });
                }
            }
        }
        // Open generation turn — model generates from here.
        buf.push_str("<|im_start|>assistant\n");
        Ok(buf)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Maps a `GenerateOutput` to the OpenAI `finish_reason` string.
    ///
    /// Returns `"stop"` when the library explicitly ended generation via a stop
    /// condition (EOS token, stop-token-id, or stop-string match); `"length"` when
    /// the token budget was exhausted without a stop condition.
    pub(super) fn finish_reason_for(
        output: &lattice_inference::model::qwen35_config::GenerateOutput,
    ) -> &'static str {
        if output.stopped { "stop" } else { "length" }
    }

    // Handlers
    // -----------------------------------------------------------------------

    pub async fn health() -> Json<HealthResponse> {
        Json(HealthResponse { status: "ok" })
    }

    pub async fn chat_completions(
        State(state): State<AppState>,
        result: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
    ) -> Result<Response, ApiError> {
        // Surface JSON extraction failures as structured 400 responses.
        // Log the raw parser message server-side; never forward it to clients.
        let Json(req) = result.map_err(|rejection| {
            if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                ApiError::PayloadTooLarge {
                    message: "request body exceeds 1 MiB limit".to_string(),
                }
            } else {
                eprintln!("invalid request body: {}", rejection.body_text());
                ApiError::BadRequest {
                    message: "invalid JSON request body".to_string(),
                    code: "invalid_request_body",
                }
            }
        })?;

        // Reject unsupported OpenAI features before any further processing.
        reject_unsupported(&req)?;

        // Validate that the caller targets the served model.
        if req.model != state.model_id {
            return Err(ApiError::BadRequest {
                message: format!(
                    "model '{}' is not loaded; this server serves '{}'",
                    req.model, state.model_id
                ),
                code: "model_not_found",
            });
        }

        if req.messages.is_empty() {
            return Err(ApiError::BadRequest {
                message: "messages must not be empty".to_string(),
                code: "invalid_messages",
            });
        }

        // Require the conversation to end with a user turn (Qwen ChatML constraint).
        let last_role = req.messages.last().map(|m| m.role.as_str()).unwrap_or("");
        if last_role != "user" {
            return Err(ApiError::BadRequest {
                message: "the last message must have role 'user'".to_string(),
                code: "invalid_messages",
            });
        }

        // Validate and resolve sampling parameters.
        let max_tokens = validate_max_tokens(
            req.max_tokens,
            req.max_completion_tokens,
            state.default_max_tokens,
            state.max_tokens_cap,
        )?;
        let temperature = validate_temperature(req.temperature)?;
        let top_p = validate_top_p(req.top_p)?;

        // Render the full conversation into a ChatML prompt.  Returns 400 for
        // any unsupported role or content-part type encountered.
        let prompt = render_prompt(&req.messages)?;

        // Preflight: reject prompts that would overflow the model's context window
        // before entering the blocking generation path.  This converts what would
        // otherwise be a panic inside spawn_blocking into a clean 400 response.
        let prompt_token_count = state.model.tokenize_len(&prompt);
        let max_context = state.model.max_context();
        if prompt_token_count == 0 || prompt_token_count.saturating_add(max_tokens) > max_context {
            return Err(ApiError::BadRequest {
                message: format!(
                    "prompt ({prompt_token_count} tokens) plus max_tokens ({max_tokens}) \
                     exceeds model context window ({max_context})"
                ),
                code: "context_length_exceeded",
            });
        }

        let stop_strings = parse_stop_strings(&req.stop)?;

        let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
            max_new_tokens: max_tokens,
            temperature,
            top_p,
            seed: req.seed,
            stop_strings,
            ..Default::default()
        };

        let model = state.model.clone();

        // Compute shared response metadata before branching on stream flag.
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let seq = state.request_counter.fetch_add(1, Ordering::Relaxed);
        let response_id = format!("chatcmpl-{created}-{seq}");

        if req.stream == Some(true) {
            // --- Streaming path ---
            //
            // `generate_streaming` is a synchronous blocking function.  We run it
            // on the blocking thread pool and feed incremental deltas into an
            // unbounded MPSC channel.  The async SSE handler drains the channel
            // and converts each message to an OpenAI `chat.completion.chunk` event.
            // An unbounded channel is acceptable here because the channel depth is
            // bounded by `max_tokens` (capped at `max_tokens_cap`): the producer
            // sends at most one `Delta` per generated token and generation halts at
            // the cap, so the worst-case buffer is a few thousand short strings —
            // the same order the non-streaming path already holds as one buffered
            // string. There is no true backpressure (an unbounded send never
            // blocks); if the client disconnects mid-stream the producer keeps
            // generating to the cap and the ignored send errors drain harmlessly.
            // Per-token backpressure / disconnect-cancellation is a future refinement.
            let (tx, rx) = futures::channel::mpsc::unbounded::<StreamMsg>();

            let stream_id = response_id.clone();
            let stream_model = state.model_id.clone();

            // Both backends funnel their result through this closure so the
            // "generated_tokens > max_tokens invariant, then finish_reason_for"
            // logic is written exactly once and shared by CPU and Metal.
            let finish_streaming = {
                let tx = tx.clone();
                move |output: GenerateOutput| {
                    if output.generated_tokens > max_tokens {
                        eprintln!(
                            "generation invariant violation: generated_tokens={} max_tokens={}",
                            output.generated_tokens, max_tokens
                        );
                        let _ = tx.unbounded_send(StreamMsg::Failed);
                    } else {
                        let finish_reason = finish_reason_for(&output);
                        let _ = tx.unbounded_send(StreamMsg::Done { finish_reason });
                    }
                }
            };

            match model {
                ModelBackend::Cpu(cpu_model) => {
                    tokio::task::spawn_blocking(move || {
                        let tx_delta = tx.clone();
                        let result = cpu_model.generate_streaming(&prompt, &gen_cfg, |delta| {
                            // Send each incremental text delta; ignore if the receiver
                            // dropped (client disconnected).
                            let _ = tx_delta.unbounded_send(StreamMsg::Delta(delta.to_string()));
                        });
                        match result {
                            Ok(output) => finish_streaming(output),
                            Err(e) => {
                                eprintln!("generation error (streaming): {e}");
                                let _ = tx.unbounded_send(StreamMsg::Failed);
                            }
                        }
                    });
                }
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { handle, .. } => {
                    tokio::spawn(async move {
                        let tx_delta = tx.clone();
                        let result = handle
                            .generate_streaming(prompt, gen_cfg, move |delta| {
                                tx_delta
                                    .unbounded_send(StreamMsg::Delta(delta.to_string()))
                                    .is_ok()
                            })
                            .await;
                        match result {
                            Ok(output) => finish_streaming(output),
                            Err(e) => {
                                eprintln!("generation error (streaming, metal): {e:?}");
                                let _ = tx.unbounded_send(StreamMsg::Failed);
                            }
                        }
                    });
                }
            }

            // Build the SSE stream.
            //
            // Event order (OpenAI spec):
            //   1. Role chunk: `delta: {"role":"assistant"}`, finish_reason absent.
            //   2. One content chunk per Delta: `delta: {"content":"..."}`, finish_reason absent.
            //   3. Finish chunk: `delta: {}`, finish_reason set.
            //   4. Literal `data: [DONE]` sentinel.
            let role_chunk = {
                let chunk = ChatCompletionChunk {
                    id: stream_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: stream_model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: Some("assistant"),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                Ok::<Event, std::convert::Infallible>(Event::default().data(data))
            };

            // Map each StreamMsg from the channel into one or two SSE events.
            let body_stream = rx.flat_map(move |msg| {
                let id = stream_id.clone();
                let mdl = stream_model.clone();
                match msg {
                    StreamMsg::Delta(text) => {
                        let chunk = ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model: mdl,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let events: Vec<Result<Event, std::convert::Infallible>> =
                            vec![Ok(Event::default().data(data))];
                        futures::stream::iter(events)
                    }
                    StreamMsg::Done { finish_reason } => {
                        let chunk = ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model: mdl,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(finish_reason),
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let events: Vec<Result<Event, std::convert::Infallible>> = vec![
                            Ok(Event::default().data(data)),
                            Ok(Event::default().data("[DONE]")),
                        ];
                        futures::stream::iter(events)
                    }
                    StreamMsg::Failed => {
                        // Emit a finish chunk with reason "stop" so the client
                        // receives a well-formed termination, then the [DONE]
                        // sentinel.  The error was already logged in the producer.
                        let chunk = ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model: mdl,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some("stop"),
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let events: Vec<Result<Event, std::convert::Infallible>> = vec![
                            Ok(Event::default().data(data)),
                            Ok(Event::default().data("[DONE]")),
                        ];
                        futures::stream::iter(events)
                    }
                }
            });

            let sse_stream = futures::stream::once(async move { role_chunk }).chain(body_stream);

            Ok(Sse::new(sse_stream)
                .keep_alive(KeepAlive::default())
                .into_response())
        } else {
            // --- Non-streaming path (CPU leg byte-identical to the original) ---
            let output = match model {
                ModelBackend::Cpu(cpu_model) => {
                    // `generate` is CPU-bound blocking work; run it on the blocking thread pool.
                    tokio::task::spawn_blocking(move || cpu_model.generate(&prompt, &gen_cfg))
                        .await
                        .map_err(|e| {
                            eprintln!("task join error: {e}");
                            ApiError::Internal {
                                message: "inference failed".to_string(),
                            }
                        })?
                        .map_err(|e| {
                            eprintln!("generation error: {e}");
                            ApiError::Internal {
                                message: "inference failed".to_string(),
                            }
                        })?
                }
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { handle, .. } => handle
                    .generate_streaming(prompt, gen_cfg, |_delta| true)
                    .await
                    .map_err(|e| {
                        eprintln!("generation error (metal): {e:?}");
                        ApiError::Internal {
                            message: "inference failed".to_string(),
                        }
                    })?,
            };

            // Distinguish "hit token cap" from "natural stop" (EOS / stop token / stop string).
            // `GenerateOutput.stopped` carries the explicit stop reason set by the library.
            // Log and return 500 if the invariant is violated.
            if output.generated_tokens > max_tokens {
                eprintln!(
                    "generation invariant violation: generated_tokens={} max_tokens={}",
                    output.generated_tokens, max_tokens
                );
                return Err(ApiError::Internal {
                    message: "inference failed".to_string(),
                });
            }
            let finish_reason = finish_reason_for(&output);

            let response = ChatCompletionResponse {
                id: response_id,
                object: "chat.completion".to_string(),
                created,
                model: state.model_id.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: output.text.clone(),
                    },
                    finish_reason: finish_reason.to_string(),
                }],
                usage: Usage {
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.generated_tokens,
                    total_tokens: output.prompt_tokens + output.generated_tokens,
                },
            };

            Ok(Json(response).into_response())
        }
    }

    // -----------------------------------------------------------------------
    // Router
    // -----------------------------------------------------------------------

    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/chat/completions", post(chat_completions))
            .layer(DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES))
            .with_state(state)
    }

    // -----------------------------------------------------------------------
    // Tests — pure helper functions; no model construction needed
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn validate_max_tokens_rejects_zero() {
            let err = validate_max_tokens(Some(0), None, 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_max_tokens",
                    ..
                }
            ));
        }

        #[test]
        fn validate_max_tokens_rejects_above_cap() {
            let err = validate_max_tokens(Some(9999), None, 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "max_tokens_exceeds_limit",
                    ..
                }
            ));
        }

        #[test]
        fn validate_max_tokens_uses_default_when_absent() {
            assert_eq!(validate_max_tokens(None, None, 128, 4096).unwrap(), 128);
        }

        #[test]
        fn validate_max_tokens_alias_agrees() {
            assert_eq!(
                validate_max_tokens(Some(512), Some(512), 256, 4096).unwrap(),
                512
            );
        }

        #[test]
        fn validate_max_tokens_alias_conflict_rejected() {
            let err = validate_max_tokens(Some(100), Some(200), 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_request",
                    ..
                }
            ));
        }

        #[test]
        fn validate_temperature_rejects_negative() {
            let err = validate_temperature(Some(-0.1)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_temperature",
                    ..
                }
            ));
        }

        #[test]
        fn validate_temperature_rejects_above_two() {
            let err = validate_temperature(Some(2.1)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_temperature",
                    ..
                }
            ));
        }

        #[test]
        fn validate_temperature_accepts_boundary() {
            assert_eq!(validate_temperature(Some(0.0)).unwrap(), 0.0);
            assert_eq!(validate_temperature(Some(2.0)).unwrap(), 2.0);
        }

        #[test]
        fn validate_top_p_rejects_zero() {
            let err = validate_top_p(Some(0.0)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_top_p",
                    ..
                }
            ));
        }

        #[test]
        fn validate_top_p_rejects_above_one() {
            let err = validate_top_p(Some(1.1)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_top_p",
                    ..
                }
            ));
        }

        #[test]
        fn validate_top_p_accepts_one() {
            assert_eq!(validate_top_p(Some(1.0)).unwrap(), 1.0);
        }

        #[test]
        fn render_prompt_multi_message_chatml() {
            let messages = vec![
                Message {
                    role: "system".to_string(),
                    content: MessageContent::Text("Be helpful.".to_string()),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("Hello".to_string()),
                },
            ];
            let prompt = render_prompt(&messages).unwrap();
            assert!(prompt.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
            assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
        }

        #[test]
        fn render_prompt_rejects_invalid_role() {
            let messages = vec![Message {
                role: "function".to_string(),
                content: MessageContent::Text("data".to_string()),
            }];
            let err = render_prompt(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_role",
                    ..
                }
            ));
        }

        #[test]
        fn render_prompt_rejects_tool_role() {
            let messages = vec![Message {
                role: "tool".to_string(),
                content: MessageContent::Text("result".to_string()),
            }];
            let err = render_prompt(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn render_prompt_rejects_non_text_content_part() {
            let messages = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![ContentPart {
                    kind: "image_url".to_string(),
                    text: None,
                }]),
            }];
            let err = render_prompt(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        // Exercises finish_reason_for via the real helper function used by the handler.
        // A cap-reached output has stopped=false → "length".
        // A stop-condition output has stopped=true → "stop".
        #[test]
        fn finish_reason_length_only_at_cap() {
            use lattice_inference::model::qwen35_config::GenerateOutput;
            let cap = GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: 10,
                generated_tokens: 64,
                stopped: false,
                stop_reason: Some(lattice_inference::StopReason::Length),
            };
            assert_eq!(super::finish_reason_for(&cap), "length");

            let natural = GenerateOutput {
                text: "hello".into(),
                token_ids: vec![1, 2, 3],
                prompt_tokens: 10,
                generated_tokens: 3,
                stopped: true,
                stop_reason: Some(lattice_inference::StopReason::Eos),
            };
            assert_eq!(super::finish_reason_for(&natural), "stop");
        }

        // M1 regression: a stop-string hit at exactly max_new_tokens must yield "stop",
        // not "length". The old token-count formula (generated == cap → "length") would
        // mislabel this case because the stop-completing token is included in generated_ids
        // before the stop is detected.
        //
        // This test calls the real finish_reason_for helper. It is RED when
        // finish_reason_for reverts to the old `generated_tokens == max_tokens` formula.
        #[test]
        fn finish_reason_stop_string_at_cap_is_stop_not_length() {
            use lattice_inference::model::qwen35_config::GenerateOutput;
            let max_tokens: usize = 4;
            // stop-string hit at exactly the token budget:
            // stopped=true because a stop string matched; generated_tokens==max_tokens
            // because the matching token is included in generated_ids before truncation.
            let output = GenerateOutput {
                text: "hi".into(),
                token_ids: vec![1, 2, 3, 4],
                prompt_tokens: 5,
                generated_tokens: max_tokens,
                stopped: true,
                stop_reason: Some(lattice_inference::StopReason::Eos),
            };
            assert_eq!(
                super::finish_reason_for(&output),
                "stop",
                "stop-string hit at cap must yield finish_reason=stop, not length"
            );
        }

        // Natural length cap (no stop condition) must still yield "length".
        #[test]
        fn finish_reason_natural_length_cap_is_length() {
            use lattice_inference::model::qwen35_config::GenerateOutput;
            let output = GenerateOutput {
                text: "hi".into(),
                token_ids: vec![1, 2, 3, 4],
                prompt_tokens: 5,
                generated_tokens: 4,
                stopped: false,
                stop_reason: Some(lattice_inference::StopReason::Length),
            };
            assert_eq!(super::finish_reason_for(&output), "length");
        }

        #[test]
        fn reject_unsupported_stream_true_ok() {
            // stream=true is now handled by the streaming path and must NOT be
            // rejected by reject_unsupported.
            let req = ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: Some(true),
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: None,
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        // -----------------------------------------------------------------------
        // ChatCompletionChunk serialization
        // -----------------------------------------------------------------------

        #[test]
        fn chunk_content_delta_serializes_correctly() {
            let chunk = ChatCompletionChunk {
                id: "chatcmpl-1-0".to_string(),
                object: "chat.completion.chunk",
                created: 1_000_000,
                model: "test-model".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: Some("Hello".to_string()),
                    },
                    finish_reason: None,
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap();
            assert!(
                json.contains("\"object\":\"chat.completion.chunk\""),
                "must contain object field"
            );
            assert!(
                json.contains("\"delta\":{\"content\":\"Hello\"}"),
                "delta must contain only content when role is None"
            );
            // finish_reason must be absent (not null) when None
            assert!(
                !json.contains("finish_reason"),
                "finish_reason must be omitted when None"
            );
        }

        #[test]
        fn chunk_finish_delta_serializes_correctly() {
            let chunk = ChatCompletionChunk {
                id: "chatcmpl-1-0".to_string(),
                object: "chat.completion.chunk",
                created: 1_000_000,
                model: "test-model".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop"),
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap();
            assert!(
                json.contains("\"finish_reason\":\"stop\""),
                "finish chunk must include finish_reason"
            );
            // delta should be empty object since both role and content are None
            assert!(
                json.contains("\"delta\":{}"),
                "finish chunk delta must be empty object"
            );
        }

        #[test]
        fn reject_unsupported_n_gt_1() {
            let req = ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: Some(3),
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_response_format_json() {
            let req = ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: Some(ResponseFormat {
                    r#type: "json_object".to_string(),
                }),
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: None,
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        // -----------------------------------------------------------------------
        // reject_unsupported — remaining fields
        // -----------------------------------------------------------------------

        fn bare_req() -> ChatCompletionRequest {
            ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: None,
            }
        }

        #[test]
        fn reject_unsupported_tools_rejected() {
            let req = ChatCompletionRequest {
                tools: Some(serde_json::json!([])),
                ..bare_req()
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_tool_choice_rejected() {
            let req = ChatCompletionRequest {
                tool_choice: Some(serde_json::json!("auto")),
                ..bare_req()
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_logprobs_rejected() {
            let req = ChatCompletionRequest {
                logprobs: Some(true),
                ..bare_req()
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_stop_now_accepted() {
            // stop is no longer rejected by reject_unsupported; it is parsed separately.
            let req = ChatCompletionRequest {
                stop: Some(serde_json::json!("</s>")),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        // -----------------------------------------------------------------------
        // parse_stop_strings
        // -----------------------------------------------------------------------

        #[test]
        fn parse_stop_strings_null_gives_empty() {
            assert_eq!(parse_stop_strings(&None).unwrap(), Vec::<String>::new());
            assert_eq!(
                parse_stop_strings(&Some(serde_json::Value::Null)).unwrap(),
                Vec::<String>::new()
            );
        }

        #[test]
        fn parse_stop_strings_single_string_gives_vec_of_one() {
            let v = parse_stop_strings(&Some(serde_json::json!("</s>"))).unwrap();
            assert_eq!(v, vec!["</s>".to_string()]);
        }

        #[test]
        fn parse_stop_strings_array_of_two_accepted() {
            let v = parse_stop_strings(&Some(serde_json::json!(["</s>", "\nUser:"]))).unwrap();
            assert_eq!(v, vec!["</s>".to_string(), "\nUser:".to_string()]);
        }

        #[test]
        fn parse_stop_strings_empty_array_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!([]))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_array_over_four_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d", "e"])))
                .unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_array_with_number_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(["ok", 42]))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_empty_string_element_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(["ok", ""]))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_empty_string_scalar_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(""))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_array_exactly_four_accepted() {
            let v = parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d"]))).unwrap();
            assert_eq!(v.len(), 4);
        }

        #[test]
        fn reject_unsupported_stream_false_ok() {
            // stream=false must not trigger a rejection.
            let req = ChatCompletionRequest {
                stream: Some(false),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        #[test]
        fn reject_unsupported_n_1_ok() {
            let req = ChatCompletionRequest {
                n: Some(1),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        #[test]
        fn reject_unsupported_response_format_text_ok() {
            let req = ChatCompletionRequest {
                response_format: Some(ResponseFormat {
                    r#type: "text".to_string(),
                }),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        #[test]
        fn reject_unsupported_logprobs_false_ok() {
            let req = ChatCompletionRequest {
                logprobs: Some(false),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        // -----------------------------------------------------------------------
        // validate_max_tokens — additional edge cases
        // -----------------------------------------------------------------------

        #[test]
        fn validate_max_tokens_at_exactly_cap_ok() {
            assert_eq!(
                validate_max_tokens(Some(4096), None, 256, 4096).unwrap(),
                4096
            );
        }

        #[test]
        fn validate_max_tokens_max_completion_only_ok() {
            assert_eq!(
                validate_max_tokens(None, Some(512), 256, 4096).unwrap(),
                512
            );
        }

        // -----------------------------------------------------------------------
        // validate_temperature — default path
        // -----------------------------------------------------------------------

        #[test]
        fn validate_temperature_none_uses_default() {
            assert_eq!(validate_temperature(None).unwrap(), 0.7);
        }

        // -----------------------------------------------------------------------
        // validate_top_p — default path
        // -----------------------------------------------------------------------

        #[test]
        fn validate_top_p_none_uses_default() {
            assert_eq!(validate_top_p(None).unwrap(), 0.9);
        }

        // -----------------------------------------------------------------------
        // render_prompt — additional cases
        // -----------------------------------------------------------------------

        #[test]
        fn render_prompt_user_only() {
            let msgs = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
            }];
            let prompt = render_prompt(&msgs).unwrap();
            assert_eq!(
                prompt,
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
            );
        }

        #[test]
        fn render_prompt_multi_turn_assistant() {
            let msgs = vec![
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("q1".to_string()),
                },
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text("a1".to_string()),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("q2".to_string()),
                },
            ];
            let prompt = render_prompt(&msgs).unwrap();
            assert!(prompt.contains("<|im_start|>user\nq1<|im_end|>"));
            assert!(prompt.contains("<|im_start|>assistant\na1<|im_end|>"));
            assert!(prompt.contains("<|im_start|>user\nq2<|im_end|>"));
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
        }

        #[test]
        fn render_prompt_content_parts_text_ok() {
            let msgs = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![
                    ContentPart {
                        kind: "text".to_string(),
                        text: Some("hello".to_string()),
                    },
                    ContentPart {
                        kind: "text".to_string(),
                        text: Some(" world".to_string()),
                    },
                ]),
            }];
            let prompt = render_prompt(&msgs).unwrap();
            assert!(prompt.contains("<|im_start|>user\nhello world<|im_end|>"));
        }

        #[test]
        fn render_prompt_rejects_developer_role() {
            let msgs = vec![Message {
                role: "developer".to_string(),
                content: MessageContent::Text("system prompt".to_string()),
            }];
            let err = render_prompt(&msgs).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        // -----------------------------------------------------------------------
        // Error envelope JSON shape
        // -----------------------------------------------------------------------

        #[test]
        fn error_envelope_bad_request_shape() {
            let err = ApiError::BadRequest {
                message: "test error".to_string(),
                code: "invalid_request",
            };
            // Verify the error serialises to the OpenAI envelope shape:
            // {"error":{"message":"...","type":"invalid_request_error","code":"...","param":null}}
            let body = ErrorBody {
                error: ErrorDetail {
                    message: "test error".to_string(),
                    r#type: "invalid_request_error",
                    code: "invalid_request".to_string(),
                    param: None,
                },
            };
            let json = serde_json::to_string(&body).unwrap();
            assert!(json.contains("\"error\""));
            assert!(json.contains("\"message\":\"test error\""));
            assert!(json.contains("\"type\":\"invalid_request_error\""));
            assert!(json.contains("\"code\":\"invalid_request\""));
            assert!(json.contains("\"param\":null"));
            // Ensure it is NOT a bare message — must be nested under "error".
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(v["error"].is_object(), "top-level key must be 'error'");
            // Variant check kept separate so we know err itself was constructed correctly.
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_request",
                    ..
                }
            ));
        }

        #[test]
        fn error_envelope_payload_too_large_shape() {
            let body = ErrorBody {
                error: ErrorDetail {
                    message: "request body exceeds 1 MiB limit".to_string(),
                    r#type: "invalid_request_error",
                    code: "request_body_too_large".to_string(),
                    param: None,
                },
            };
            let json = serde_json::to_string(&body).unwrap();
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["error"]["code"], "request_body_too_large");
        }

        #[test]
        fn error_envelope_internal_shape() {
            let body = ErrorBody {
                error: ErrorDetail {
                    message: "inference failed".to_string(),
                    r#type: "server_error",
                    code: "internal_error".to_string(),
                    param: None,
                },
            };
            let json = serde_json::to_string(&body).unwrap();
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["error"]["type"], "server_error");
            assert_eq!(v["error"]["code"], "internal_error");
        }

        // -----------------------------------------------------------------------
        // message_text helper
        // -----------------------------------------------------------------------

        #[test]
        fn message_text_plain_string() {
            let content = MessageContent::Text("hello".to_string());
            assert_eq!(message_text(&content).unwrap(), "hello");
        }

        #[test]
        fn message_text_parts_concatenates() {
            let content = MessageContent::Parts(vec![
                ContentPart {
                    kind: "text".to_string(),
                    text: Some("foo".to_string()),
                },
                ContentPart {
                    kind: "text".to_string(),
                    text: Some("bar".to_string()),
                },
            ]);
            assert_eq!(message_text(&content).unwrap(), "foobar");
        }

        #[test]
        fn message_text_parts_rejects_image() {
            let content = MessageContent::Parts(vec![ContentPart {
                kind: "image_url".to_string(),
                text: None,
            }]);
            let err = message_text(&content).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Chat {
            model,
            max_tokens,
            temperature,
            tokenizer_dir,
        } => {
            run_chat(&model, max_tokens, temperature, tokenizer_dir.as_deref());
        }
        Command::Serve {
            model,
            host,
            port,
            max_tokens,
            model_id,
            tokenizer_dir,
        } => {
            use std::path::Path;
            use std::sync::Arc;
            use std::sync::atomic::AtomicU64;

            // Derive a model identifier from the path basename when --model-id
            // is not provided.
            let served_model_id = model_id.unwrap_or_else(|| {
                Path::new(&model)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("lattice")
                    .to_string()
            });

            let model_path = Path::new(&model);
            let format = backend::detect_format(model_path);

            eprintln!("Loading model from {model}...");
            let model_backend: serve::ModelBackend = match format {
                backend::ModelFormat::Safetensors => {
                    match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(
                        model_path,
                    ) {
                        Ok(m) => serve::ModelBackend::Cpu(Arc::new(m)),
                        Err(e) => {
                            eprintln!("Error: failed to load model: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                backend::ModelFormat::Q4 => {
                    #[cfg(feature = "metal-gpu")]
                    {
                        let tokenizer_dir_path =
                            tokenizer_dir.as_ref().map(std::path::PathBuf::from);
                        match serve::ModelBackend::spawn_metal(
                            model_path.to_path_buf(),
                            tokenizer_dir_path,
                        ) {
                            Ok((backend, _max_context)) => backend,
                            Err(e) => {
                                eprintln!("Error: failed to load Q4 model: {e}");
                                std::process::exit(1);
                            }
                        }
                    }
                    #[cfg(not(feature = "metal-gpu"))]
                    {
                        let _ = &tokenizer_dir;
                        eprintln!("Error: {}", backend::metal_gpu_required_message(model_path));
                        std::process::exit(1);
                    }
                }
                backend::ModelFormat::Unknown => {
                    eprintln!(
                        "Error: {}",
                        backend::unrecognized_format_message(model_path)
                    );
                    std::process::exit(1);
                }
            };
            eprintln!("Model loaded. Serving as '{served_model_id}'.");

            let state = serve::AppState {
                model: model_backend,
                default_max_tokens: max_tokens,
                max_tokens_cap: 4096,
                model_id: served_model_id.clone(),
                request_counter: Arc::new(AtomicU64::new(0)),
            };

            let app = serve::router(state);

            let addr = format!("{host}:{port}");
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error: failed to bind to {addr}: {e}");
                    std::process::exit(1);
                }
            };
            eprintln!(
                "Listening on {addr}  (model: {served_model_id}, max_tokens default: {max_tokens})"
            );
            eprintln!("  POST /v1/chat/completions");
            eprintln!("  GET  /health");

            let shutdown = async {
                if let Err(e) = tokio::signal::ctrl_c().await {
                    eprintln!("Error waiting for shutdown signal: {e}");
                }
                eprintln!("Shutdown signal received, draining connections...");
            };

            if let Err(e) = axum::serve(listener, app)
                .with_graceful_shutdown(shutdown)
                .await
            {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
        Command::Doctor {
            model,
            context,
            tokenizer_dir,
        } => {
            use std::path::Path;

            let model_path = Path::new(&model);
            let tokenizer_dir_path = tokenizer_dir.as_deref().map(Path::new);
            match doctor::build_report(model_path, tokenizer_dir_path, context, None) {
                Ok(report) => {
                    println!("{report}");
                    if !report.is_ready() {
                        eprintln!("doctor: model is NOT usable as configured (see reasons above)");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
