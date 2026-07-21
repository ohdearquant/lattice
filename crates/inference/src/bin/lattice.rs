//! `lattice` CLI - interactive chat, HTTP serve, and preflight subcommands. See [docs/capability-matrix.md](../../../../docs/capability-matrix.md).
//!
//! # Usage
//!
//! ```text
//! lattice chat --model /path/to/model [--max-tokens 256] [--temperature 0.7]
//! lattice serve --model /path/to/model [--host 127.0.0.1] [--port 8080] [--max-tokens 256]
//! lattice doctor --model /path/to/model [--context 4096]
//! lattice prune-score --q4-dir /path/to/model-q4 --tokenizer-dir /path/to/model \
//!   --calibration-corpus calibration.txt --validation-corpus validation.txt \
//!   --prune-layers 4 --output lattice_pruning.json
//! ```

use clap::{Parser, Subcommand};

#[path = "lattice/prune_score.rs"]
mod prune_score;

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
        /// Cap on outstanding (queued + in-flight) requests to the Metal GPU
        /// worker (issue #932) before new requests are rejected with HTTP
        /// 503. Only applies to Q4/Metal-backed serving; the CPU backend has
        /// no shared worker queue to bound. Conservative default: this
        /// worker serializes all generation onto one dedicated thread, so a
        /// deep queue just means memory growth with no throughput benefit.
        /// Must be between 1 and `tokio::sync::Semaphore::MAX_PERMITS`
        /// (issue #939): zero would admit nothing (every request fails
        /// admission), and clap rejects anything larger here instead of
        /// deferring to `MetalWorker::spawn`'s own
        /// `Semaphore::new`-precondition panic.
        #[arg(
            long,
            default_value = "32",
            value_parser = clap::builder::RangedU64ValueParser::<usize>::new()
                .range(1..=(tokio::sync::Semaphore::MAX_PERMITS as u64))
        )]
        max_pending: usize,
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
    /// Score layer importance on a calibration corpus and PPL-gate a pruning plan.
    PruneScore {
        #[command(flatten)]
        args: prune_score::Args,
    },
}

// ─── #939 CLI boundary tests: `--max-pending` range validation ────────────
//
// clap's own `value_parser!(usize).range(1..=Semaphore::MAX_PERMITS)` on the
// `Serve::max_pending` field (above) is the ONLY validation this binary
// needs for zero / too-large values -- unlike `lattice_serve.rs`'s hand
// rolled argv parser, clap already rejects a malformed string (`abc`,
// `-1`) itself, before this range check ever runs. These tests exercise
// that `value_parser` wiring directly through `Cli::try_parse_from`,
// rather than duplicating the range logic anywhere in this binary.
#[cfg(test)]
mod max_pending_cli_tests {
    use super::*;

    fn parse_max_pending(args: &[&str]) -> Result<usize, clap::Error> {
        let mut full = vec!["lattice", "serve", "--model", "/tmp/model"];
        full.extend_from_slice(args);
        match Cli::try_parse_from(full)?.command {
            Command::Serve { max_pending, .. } => Ok(max_pending),
            _ => panic!("expected Command::Serve, got a different Command variant"),
        }
    }

    #[test]
    fn max_pending_omitted_defaults_to_32() {
        assert_eq!(parse_max_pending(&[]).expect("no --max-pending"), 32);
    }

    #[test]
    fn max_pending_zero_is_rejected() {
        parse_max_pending(&["--max-pending", "0"])
            .expect_err("0 admits nothing and must be rejected, not silently accepted");
    }

    #[test]
    fn max_pending_one_above_max_permits_is_rejected() {
        let too_big = (tokio::sync::Semaphore::MAX_PERMITS as u128 + 1).to_string();
        parse_max_pending(&["--max-pending", &too_big]).expect_err(
            "Semaphore::MAX_PERMITS + 1 must be rejected before it can panic Semaphore::new",
        );
    }

    #[test]
    fn max_pending_at_max_permits_is_accepted() {
        let at_max = tokio::sync::Semaphore::MAX_PERMITS.to_string();
        assert_eq!(
            parse_max_pending(&["--max-pending", &at_max])
                .expect("Semaphore::MAX_PERMITS itself is the inclusive upper bound"),
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }

    #[test]
    fn max_pending_negative_is_rejected() {
        parse_max_pending(&["--max-pending", "-1"])
            .expect_err("a negative value must be rejected, not silently defaulted");
    }

    #[test]
    fn max_pending_malformed_is_rejected() {
        parse_max_pending(&["--max-pending", "not-a-number"])
            .expect_err("a non-numeric value must be rejected, not silently defaulted");
    }

    #[test]
    fn max_pending_valid_override_is_accepted() {
        assert_eq!(
            parse_max_pending(&["--max-pending", "8"]).expect("8 is a valid cap"),
            8
        );
    }
}

// ---------------------------------------------------------------------------
// backend: model-directory format detection + Q4/Metal loading
//
// `lattice chat`/`lattice serve` originally only understood a safetensors
// directory (`model.safetensors` or a sharded index). Native Q4 quantized
// directories (per-tensor `.q4` files, the output of `quantize_q4`) route to
// the Metal GPU forward pass instead. Safetensors directories are completely
// unaffected: `detect_format` returns `Safetensors` for them exactly as
// before, and the safetensors load path is untouched.
//
// The detector itself (`ModelFormat` + `detect_format` + the two error
// message helpers) now lives in `lattice_inference::model_format` (ADR-080
// amendment, #829): it is shared, unmodified, with `lattice_serve.rs` and
// `chat_metal.rs`, which cannot see a `pub(crate)` item defined in this
// binary's own crate root. `backend` here is a local alias so every existing
// `backend::...` / `crate::backend::...` call site below is unchanged.
// ---------------------------------------------------------------------------

use lattice_inference::model_format as backend;

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
                let Ok(file_path) =
                    lattice_inference::weights::contained_shard_path(dir, &entry.file)
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
            let threshold_override =
                corrected_total + (stale_would_have_been - corrected_total) / 2;
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
}

// ---------------------------------------------------------------------------
// chat subcommand
// ---------------------------------------------------------------------------

/// Load `config.json` for a Q4 directory, via the single shared
/// config-resolution policy (`Qwen35Config::from_model_dir`, #923) used by
/// every loader in this crate: a missing `config.json` is a hard,
/// descriptive error naming the directory, never a silently-substituted
/// architecture preset.
#[cfg(feature = "metal-gpu")]
fn load_q4_config(
    dir: &std::path::Path,
) -> Result<lattice_inference::model::qwen35_config::Qwen35Config, String> {
    lattice_inference::model::qwen35_config::Qwen35Config::from_model_dir(dir)
        .map_err(|e| format!("config.json load failed: {e}"))
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
    ) -> Result<
        lattice_inference::model::qwen35_config::GenerateOutput,
        lattice_inference::error::InferenceError,
    > {
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
            Backend::Metal(m) => match m.generate(trimmed, &gen_cfg) {
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
        response::{
            IntoResponse, Response,
            sse::{Event, KeepAlive, Sse},
        },
        routing::{get, post},
    };
    use futures::StreamExt as _;
    use lattice_inference::Tokenizer;
    // `ChatMessage` and `format_chat_template` are CPU-available (#668): this
    // binary's own CPU (safetensors) serve path renders every request's
    // ChatML prompt through them (`prepare_chat_request`, via the shared
    // `serve::contract::normalize_messages`), the same shared renderer the
    // Metal worker loop uses -- there is no second bespoke ChatML renderer
    // in this file anymore. `to_chat_messages` below is a `#[cfg(test)]`-only
    // helper local to this binary's own tests, not part of that production
    // path.
    use lattice_inference::forward::metal_qwen35::{ChatMessage, format_chat_template};
    #[cfg(feature = "metal-gpu")]
    use lattice_inference::model::qwen35_config::GenerateConfig;
    use lattice_inference::model::qwen35_config::{GenerateOutput, TokenLogprob};
    use lattice_inference::serve::contract::{
        ChatRequest as ChatCompletionRequest, GenerationDefaults, ServeProfile,
        ValidatedChatRequest as ContractValidatedChatRequest, normalize_request,
        validate_context_window,
    };
    #[cfg(test)]
    use lattice_inference::serve::contract::{
        ContentPart, Message, MessageContent, ResponseFormat,
    };
    use serde::Serialize;
    use serde_json::Value;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Request body cap: 1 MiB.  Requests above this return HTTP 413.
    /// ADR-080 C2 (#782): `lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES`
    /// is the single shared constant now; both binaries previously carried
    /// this exact value independently.
    use lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES;

    // -----------------------------------------------------------------------
    // Model backend: CPU (safetensors) or Metal GPU (native Q4)
    // -----------------------------------------------------------------------

    /// Handle to the shared Metal GPU worker thread
    /// (`lattice_inference::serve::metal_worker::MetalWorker`, issue #832:
    /// the single shared owner module that replaces this binary's prior
    /// private `MetalJob`/`MetalHandle` loop and `lattice_serve.rs`'s prior
    /// private `Job`/`spawn_worker`/`run_worker_loop` loop). Cheaply `Clone`
    /// (wraps a `MetalWorkerClient`, itself backed by an `mpsc` sender),
    /// `Send + Sync`, so it can live in `AppState` like the CPU
    /// `Arc<Qwen35Model>` does — only the underlying `MetalQwen35State`
    /// inside the shared worker thread is confined to that thread.
    ///
    /// This still serializes ALL Metal generation onto one thread: two
    /// concurrent requests to a Q4-backed `lattice serve` run back-to-back,
    /// not in parallel. That is correct for a single-GPU local engine (the
    /// same default ollama uses) and is documented here rather than hidden
    /// behind an innocuous-looking channel send.
    #[cfg(feature = "metal-gpu")]
    #[derive(Clone)]
    pub struct MetalHandle {
        client: lattice_inference::serve::metal_worker::MetalWorkerClient,
    }

    #[cfg(feature = "metal-gpu")]
    impl MetalHandle {
        /// Normalizes
        /// [`lattice_inference::serve::metal_worker::WorkerEvent::Cancelled`]
        /// (the job was skipped at dequeue time: the client's `cancel`
        /// watch flag was already `true`, or this job's own event receiver
        /// was already closed) into the exact empty, interrupted
        /// `GenerateOutput` this binary's prior dequeue-cancellation reply
        /// already produced (#832: preserves this binary's own pre-existing
        /// observable shape — the shared `WorkerEvent::Cancelled` contract
        /// itself was modeled on it; see the shared module's own doc
        /// comment).
        fn normalize_cancelled(
            ev: lattice_inference::serve::metal_worker::WorkerEvent,
        ) -> lattice_inference::serve::metal_worker::WorkerEvent {
            use lattice_inference::serve::metal_worker::WorkerEvent;
            match ev {
                WorkerEvent::Cancelled => WorkerEvent::Complete(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: 0,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(lattice_inference::StopReason::Interrupt),
                    token_logprobs: vec![],
                }),
                other => other,
            }
        }

        /// Run one generation on the shared worker thread, forwarding each
        /// token delta to `on_token`. Returns the full `GenerateOutput`
        /// (including `stopped`/`stop_reason`) so callers can compute
        /// `finish_reason` with the exact same `finish_reason_for` helper
        /// the CPU path uses.
        ///
        /// Returns `Err` if the worker thread is unreachable
        /// (`ApiError::Internal`, "inference worker unavailable" — the same
        /// wording `lattice_serve.rs` uses for the identical condition on
        /// this shared worker contract, #832; this binary's prior distinct
        /// "not running" vs "dropped the request" phrasing collapses into
        /// one message here, since `MetalWorkerClient::submit` no longer
        /// exposes that distinction to callers), if the request cannot fit
        /// the model's context window (`ApiError::BadRequest`, surfaced by
        /// the shared worker's `check_prompt_fits_window` — new coverage
        /// for this binary; the HTTP-layer `check_context_window` preflight
        /// in `prepare_chat_request` already rejects the overwhelming
        /// majority of these before this call is ever reached), or if the
        /// underlying `generate_streaming` call itself fails closed (#611:
        /// e.g. a grammar mask that blocks every candidate token) —
        /// collapsed to `ApiError::Internal` here, matching the same
        /// generic "inference failed" 500 the CPU path already returns for
        /// any `generate()` error.
        async fn generate_streaming(
            &self,
            messages: Vec<ChatMessage>,
            gen_cfg: GenerateConfig,
            on_token: impl FnMut(&str) -> bool + Send + 'static,
        ) -> Result<GenerateOutput, ApiError> {
            // `cancel = never-fires` convenience form of
            // `generate_streaming_with_cancel`, for callers that do not wire
            // up disconnect cancellation.
            let (_never_cancels, cancel_rx) = tokio::sync::watch::channel(false);
            self.generate_streaming_with_cancel(messages, gen_cfg, on_token, cancel_rx)
                .await
        }

        /// Cancellation-aware sibling of [`Self::generate_streaming`]
        /// (ADR-080 C2, #744): `cancel` starts `false` and flips to `true`
        /// the moment the caller's paired
        /// `lattice_inference::serve::CancelOnDrop` guard is dropped (client
        /// disconnect). The shared worker (issue #832) checks it
        /// independently of `on_token`'s return value — before prefill,
        /// immediately after prefill, and at the top of every decode
        /// iteration — via
        /// `generate_streaming_with_prefix_cache_and_cancel`'s
        /// `should_cancel` predicate, and once more at dequeue time before
        /// paying for prefill on an already-abandoned job.
        ///
        /// `on_token`'s return value is still honored (this method stops
        /// calling it the first time it returns `false`, e.g. a
        /// disconnected `tx_delta`), but can no longer stop the worker
        /// thread directly the way it did when `MetalJob` embedded the
        /// callback inside the worker itself — the shared worker now lives
        /// behind a `WorkerEvent` channel this method drains from a
        /// separate async task, and `cancel` (checked independently by the
        /// worker) is the only signal that can reach across that boundary.
        /// In practice this is not a behavior change: `tx_delta` and the
        /// `cancel_guard` pairing `cancel` are dropped by the exact same
        /// axum stream-drop event at every call site, so `cancel` already
        /// catches a disconnect at essentially the same moment `on_token`
        /// would have.
        async fn generate_streaming_with_cancel(
            &self,
            messages: Vec<ChatMessage>,
            gen_cfg: GenerateConfig,
            on_token: impl FnMut(&str) -> bool + Send + 'static,
            cancel: tokio::sync::watch::Receiver<bool>,
        ) -> Result<GenerateOutput, ApiError> {
            // #932: the ONE way `MetalWorkerClient::submit` fails outwardly
            // -- the shared worker's outstanding-job admission cap is full.
            // Surfaces as an ordinary `ApiError::ServiceUnavailable` (503),
            // exactly like any other `Err` this method already returns.
            let mut rx = self.submit(messages, gen_cfg, cancel)?;
            Self::drain(&mut rx, on_token).await
        }

        /// Admission-only half of [`Self::generate_streaming_with_cancel`]
        /// (#939): runs `MetalWorkerClient::submit`'s synchronous admission
        /// check (issue #932's `Semaphore::try_acquire_owned`) and returns
        /// immediately, before any event is drained from the worker.
        ///
        /// Split out so `chat_completions`'s streaming arm can call this
        /// directly -- and propagate `Err(ApiError::ServiceUnavailable)`
        /// with `?` -- BEFORE building and returning the SSE response,
        /// instead of discovering admission failure only after a detached
        /// `tokio::spawn` task (which may not even have started running
        /// yet) reaches this call. Draining is [`Self::drain`], called
        /// separately once the response has been committed.
        fn submit(
            &self,
            messages: Vec<ChatMessage>,
            gen_cfg: GenerateConfig,
            cancel: tokio::sync::watch::Receiver<bool>,
        ) -> Result<
            tokio::sync::mpsc::UnboundedReceiver<
                lattice_inference::serve::metal_worker::WorkerEvent,
            >,
            ApiError,
        > {
            self.client.submit(messages, gen_cfg, cancel)
        }

        /// Drains a receiver obtained from [`Self::submit`], forwarding
        /// token deltas to `on_token` and normalizing
        /// `WorkerEvent::Cancelled` -- exactly the loop
        /// `generate_streaming_with_cancel` ran inline before admission was
        /// split out of it (#939).
        async fn drain(
            rx: &mut tokio::sync::mpsc::UnboundedReceiver<
                lattice_inference::serve::metal_worker::WorkerEvent,
            >,
            mut on_token: impl FnMut(&str) -> bool + Send + 'static,
        ) -> Result<GenerateOutput, ApiError> {
            use lattice_inference::serve::metal_worker::WorkerEvent;

            let mut deliver_deltas = true;
            loop {
                let Some(ev) = rx.recv().await else {
                    return Err(ApiError::Internal {
                        message: "inference worker unavailable".to_string(),
                    });
                };
                match Self::normalize_cancelled(ev) {
                    WorkerEvent::Delta(delta) => {
                        if deliver_deltas && !on_token(&delta) {
                            deliver_deltas = false;
                        }
                    }
                    WorkerEvent::Complete(output) => return Ok(output),
                    WorkerEvent::Rejected(api_err) => return Err(api_err),
                    WorkerEvent::Failed(message) | WorkerEvent::ConstraintBlocked(message) => {
                        return Err(ApiError::Internal {
                            message: format!("generation failed: {message}"),
                        });
                    }
                    WorkerEvent::Cancelled => {
                        unreachable!("normalize_cancelled already rewrote Cancelled into Complete")
                    }
                }
            }
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
        /// Test-only seam (ADR-080 C2): wraps a real tiny model for
        /// `tokenize_len`/`max_context`/
        /// `tokenizer` (so request validation stays realistic) but
        /// substitutes the CPU streaming generation call itself with an
        /// injected closure. This lets a test observe `should_cancel` being
        /// polled by the EXACT production composition in
        /// `chat_completions`'s CPU streaming arm -- the same `on_token`/
        /// `should_cancel` construction and `cancel_rx` wiring real
        /// requests use -- independently of a real decode loop's timing,
        /// isolating cancellation-signal wiring from `on_token`'s own
        /// failed-send stop condition (the exact ambiguity the disconnect
        /// test's mutation gap left open). Never constructible outside
        /// `--features test-utils`.
        #[cfg(all(feature = "test-utils", test))]
        CpuFakeGenerate {
            model: Arc<lattice_inference::model::qwen35::Qwen35Model>,
            #[allow(clippy::type_complexity)]
            generate: Arc<
                dyn Fn(
                        &str,
                        &lattice_inference::model::qwen35_config::GenerateConfig,
                        &mut dyn FnMut(&str) -> bool,
                        &mut dyn FnMut() -> bool,
                    )
                        -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                    + Send
                    + Sync,
            >,
        },
    }

    impl ModelBackend {
        pub fn tokenize_len(&self, text: &str) -> usize {
            match self {
                ModelBackend::Cpu(m) => m.tokenizer().tokenize(text).real_length,
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { tokenizer, .. } => tokenizer.tokenize(text).real_length,
                #[cfg(all(feature = "test-utils", test))]
                ModelBackend::CpuFakeGenerate { model, .. } => {
                    model.tokenizer().tokenize(text).real_length
                }
            }
        }

        pub fn max_context(&self) -> usize {
            match self {
                ModelBackend::Cpu(m) => m.max_context(),
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { max_context, .. } => *max_context,
                #[cfg(all(feature = "test-utils", test))]
                ModelBackend::CpuFakeGenerate { model, .. } => model.max_context(),
            }
        }

        /// Tokenizer for this backend, used to render `logprobs` token ids
        /// back into text/bytes (#585).
        pub fn tokenizer(&self) -> &lattice_inference::tokenizer::bpe::BpeTokenizer {
            match self {
                ModelBackend::Cpu(m) => m.tokenizer(),
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { tokenizer, .. } => tokenizer,
                #[cfg(all(feature = "test-utils", test))]
                ModelBackend::CpuFakeGenerate { model, .. } => model.tokenizer(),
            }
        }

        /// Load a native Q4 checkpoint on the shared Metal worker thread
        /// (`lattice_inference::serve::metal_worker::MetalWorker`, issue
        /// #832 — the same shared owner `lattice_serve.rs` uses) and return
        /// the `ModelBackend::Metal` handle plus the resolved context
        /// window, for `main()`'s `Command::Serve` startup sequence.
        #[cfg(feature = "metal-gpu")]
        pub fn spawn_metal(
            model_dir: std::path::PathBuf,
            tokenizer_dir: Option<std::path::PathBuf>,
            max_pending: usize,
        ) -> Result<(Self, usize), String> {
            use lattice_inference::serve::metal_worker::{
                ContextWindowPolicy, MetalWorker, StartupError, WorkerMetadata,
            };

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
            // #832: the shared worker's loader needs its own owned
            // tokenizer (it renders + tokenizes the prompt once per request
            // for the KV-window check, `check_prompt_fits_window`).
            // Cloned from the `Arc` above rather than re-read from disk, so
            // `tokenizer.json` is still parsed exactly once; this is a
            // single, one-time, startup-only in-memory clone, not a
            // per-request cost.
            let tokenizer_for_worker = (*tokenizer).clone();
            // Preserves this binary's pre-existing behavior exactly: the
            // context window is this fixed cap, not re-derived from
            // `state.max_context()` after loading (unlike
            // `lattice_serve.rs`'s `load_model`). Passed to the shared
            // worker's `WorkerMetadata` too, so its internal
            // `check_prompt_fits_window` invariant agrees with the
            // HTTP-layer `check_context_window` preflight that already runs
            // first in `prepare_chat_request`.
            let max_context = super::MetalChatBackend::MAX_CACHE_LEN;
            let model_dir_for_loader = model_dir.clone();
            let tokenizer_path_for_loader = tokenizer_path.clone();
            let (owner, client, _meta) = MetalWorker::spawn(
                move || {
                    let cfg = super::load_q4_config(&model_dir_for_loader)?;
                    let state =
                        lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir(
                            &model_dir_for_loader,
                            &tokenizer_path_for_loader,
                            &cfg,
                            max_context,
                        )
                        .map_err(|e| format!("Q4 model load failed: {e}"))?;
                    Ok((
                        state,
                        tokenizer_for_worker,
                        WorkerMetadata {
                            format: "q4".to_string(),
                            model_max_context: max_context,
                            context_window_policy: ContextWindowPolicy::PromptAndMaxTokens,
                        },
                    ))
                },
                max_pending,
            )
            .map_err(|e| match e {
                StartupError::Load(msg) => msg,
                // Preserves this binary's exact prior wording (distinct
                // from `MetalWorker::spawn`'s own generic
                // `StartupError::ThreadExited` `Display` text) for the same
                // condition: the worker thread exited/panicked before ever
                // sending a readiness signal.
                StartupError::ThreadExited => {
                    "Metal worker thread exited before loading finished".to_string()
                }
                // #939: clap's own `value_parser` range already rejects an
                // out-of-range `--max-pending` before this call, so this
                // arm is unreachable in practice through the CLI -- kept
                // for exhaustiveness and as defense in depth against any
                // other future `spawn_metal` caller.
                err @ StartupError::InvalidMaxPending { .. } => err.to_string(),
            })?;
            // `owner` (and the `JoinHandle` it carries) is dropped here,
            // immediately: this binary's prior `MetalHandle::spawn` never
            // captured `std::thread::spawn`'s return value either, so the
            // worker thread was already detached rather than joined.
            // Dropping a `JoinHandle` only detaches it (does not stop the
            // thread), so the worker keeps running until process exit
            // either way -- today's behavior is unchanged. Unlike
            // `lattice_serve.rs`'s `run()` (which blocks for the server's
            // whole lifetime and so has a natural place to hold the owner
            // for issue #833's future join-on-shutdown seam), this function
            // returns before `lattice serve`'s listener ever binds; #833
            // would need to thread `owner` into `AppState`/
            // `ModelBackend::Metal` instead if it wants an explicit join
            // point on this binary.
            drop(owner);
            Ok((
                ModelBackend::Metal {
                    handle: MetalHandle { client },
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
    // Error type (ADR-080 C2, #782): shared verbatim with `lattice_serve.rs`
    // via `lattice_inference::serve::ApiError` -- this binary's local
    // `ApiError`/`ErrorBody`/`ErrorDetail`/`IntoResponse` were byte-identical
    // to the shared definition, so they are gone; every existing
    // `ApiError::BadRequest { message, code }` / `PayloadTooLarge` /
    // `Internal` construction site below is unaffected (same variant names
    // and fields).
    // -----------------------------------------------------------------------

    use lattice_inference::serve::ApiError;

    // -----------------------------------------------------------------------
    // Request / response types
    // -----------------------------------------------------------------------

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
        /// Per-token log-probabilities (#585). `None` unless the request set
        /// `logprobs: true`, matching the OpenAI response shape where the
        /// field is omitted rather than `null` for a plain completion.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<ChoiceLogprobs>,
    }

    /// `choices[].logprobs` — OpenAI chat-completions logprobs envelope (#585).
    #[derive(Serialize)]
    pub struct ChoiceLogprobs {
        pub content: Vec<TokenLogprobEntry>,
    }

    /// One sampled token's log-probability, plus its top-N alternatives.
    #[derive(Serialize)]
    pub struct TokenLogprobEntry {
        pub token: String,
        pub logprob: f32,
        /// Raw UTF-8 bytes of `token`. `None` when the token id could not be
        /// resolved back to vocabulary text (should not happen for a token
        /// this server just sampled, but fails closed rather than panicking).
        pub bytes: Option<Vec<u8>>,
        pub top_logprobs: Vec<TopLogprobEntry>,
    }

    /// One alternative token considered at a sampled position.
    #[derive(Serialize)]
    pub struct TopLogprobEntry {
        pub token: String,
        pub logprob: f32,
        pub bytes: Option<Vec<u8>>,
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
    #[cfg(test)]
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
        // ADR-080 C2, #745: the zero-rejection itself is the shared contract
        // (`lattice_serve.rs`'s `build_cfg` silently clamped a
        // client-supplied `max_tokens: 0` through instead of rejecting it);
        // the cap-reject below stays local since the two binaries'
        // over-cap policies deliberately differ (this one rejects, the
        // daemon clamps to the model's context window).
        lattice_inference::serve::reject_zero_max_tokens(effective)?;
        if effective > max_tokens_cap {
            return Err(ApiError::BadRequest {
                message: format!("max_tokens {effective} exceeds server limit {max_tokens_cap}"),
                code: "max_tokens_exceeds_limit",
            });
        }
        Ok(effective)
    }

    /// Validate `temperature` is in `[0.0, 2.0]`.
    #[cfg(test)]
    fn validate_temperature(value: Option<f32>) -> Result<f32, ApiError> {
        lattice_inference::serve::contract::validate_temperature(value.unwrap_or(0.7))
    }

    /// Validate `top_p` is in `(0.0, 1.0]`.
    #[cfg(test)]
    fn validate_top_p(value: Option<f32>) -> Result<f32, ApiError> {
        lattice_inference::serve::contract::validate_top_p(value.unwrap_or(0.9))
    }

    /// Validate the `logprobs` / `top_logprobs` pair (#585) and resolve the
    /// number of alternatives to capture per token.
    ///
    /// - `logprobs` absent or `false` → `Ok(None)` (capture disabled, zero cost).
    /// - `logprobs: true`, `top_logprobs` absent → `Ok(Some(0))` (per-token
    ///   logprob only, no alternatives — matches the OpenAI default).
    /// - `logprobs: true`, `top_logprobs: Some(n)` with `0 <= n <= 20` → `Ok(Some(n))`.
    /// - `top_logprobs` set without `logprobs: true` → rejected (matches OpenAI).
    /// - `top_logprobs > 20` → rejected.
    #[cfg(test)]
    fn validate_logprobs(
        logprobs: Option<bool>,
        top_logprobs: Option<usize>,
    ) -> Result<Option<usize>, ApiError> {
        if !logprobs.unwrap_or(false) {
            if top_logprobs.is_some() {
                return Err(ApiError::BadRequest {
                    message: "top_logprobs requires logprobs: true".to_string(),
                    code: "invalid_request",
                });
            }
            return Ok(None);
        }
        let top_n = top_logprobs.unwrap_or(0);
        if top_n > 20 {
            return Err(ApiError::BadRequest {
                message: format!("top_logprobs {top_n} exceeds the maximum of 20"),
                code: "invalid_top_logprobs",
            });
        }
        Ok(Some(top_n))
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
    #[cfg(test)]
    fn parse_stop_strings(stop: &Option<Value>) -> Result<Vec<String>, ApiError> {
        lattice_inference::serve::contract::parse_stop_strings(stop)
    }

    /// Reject OpenAI fields that are parsed but not yet implemented.
    ///
    /// Note: `stream=true` is now handled by the streaming path in `chat_completions`
    /// and is intentionally NOT rejected here. `logprobs`/`top_logprobs` are
    /// implemented on the non-streaming path only (#585); combined with
    /// `stream: true` they are rejected below rather than silently ignored.
    #[cfg(test)]
    fn reject_unsupported(req: &ChatCompletionRequest) -> Result<(), ApiError> {
        if req.tools.is_some() || req.tool_choice.is_some() {
            return Err(ApiError::BadRequest {
                message: "tools and tool_choice are not supported by this server".to_string(),
                code: "unsupported_feature",
            });
        }
        if req.stream == Some(true) && req.logprobs.unwrap_or(false) {
            return Err(ApiError::BadRequest {
                message: "logprobs is not supported together with stream: true".to_string(),
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

    /// `#[cfg(test)]`-only alias for this binary's own role/content-part
    /// test fixtures below. Production requests go through
    /// `serve::contract::normalize_messages` directly (`prepare_chat_request`,
    /// then rendered via the shared `format_chat_template`, #661/#668); this
    /// exercises that same shared validator rather than a local copy of its
    /// role/content-part logic, so those tests cannot drift from production
    /// behavior.
    #[cfg(test)]
    fn to_chat_messages(messages: &[Message]) -> Result<Vec<ChatMessage>, ApiError> {
        lattice_inference::serve::contract::normalize_messages(messages)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Maps a `GenerateOutput` to the OpenAI `finish_reason` string (ADR-080
    /// C2, #746): delegates to the shared `lattice_inference::serve::
    /// finish_reason`, which both binaries now use so the mapping cannot
    /// drift between them again -- `lattice_serve.rs`'s worker previously
    /// hardcoded `"stop"` unconditionally instead of carrying the engine's
    /// `stopped` flag through.
    pub(super) fn finish_reason_for(
        output: &lattice_inference::model::qwen35_config::GenerateOutput,
    ) -> &'static str {
        lattice_inference::serve::finish_reason(output.stopped)
    }

    /// Resolve a token id back to its OpenAI `logprobs` text/bytes representation (#585).
    ///
    /// `token` uses the lossy UTF-8 rendering (matches OpenAI, which also shows
    /// replacement characters for a token that is only part of a multi-byte
    /// codepoint); `bytes` carries the exact original bytes so callers can
    /// reconstruct byte-accurate output regardless of codepoint boundaries.
    ///
    /// Every token id this server places into `token_logprobs` was just sampled
    /// by this same tokenizer's vocabulary, so `token_for_id` returning `None`
    /// is not expected in practice; the fallback fails closed with a visibly
    /// synthetic token string and no bytes, rather than panicking.
    fn render_token_logprob(
        tokenizer: &lattice_inference::tokenizer::bpe::BpeTokenizer,
        token_id: u32,
    ) -> (String, Option<Vec<u8>>) {
        match tokenizer.token_bytes_for_id(token_id) {
            Some(bytes) => (String::from_utf8_lossy(&bytes).into_owned(), Some(bytes)),
            None => (format!("<|unresolved_token_{token_id}|>"), None),
        }
    }

    /// Build the `choices[].logprobs` envelope from the engine's raw
    /// `token_logprobs` (#585). `token_logprobs` is empty when `logprobs` was
    /// not requested, in which case this returns an empty `content` — callers
    /// only invoke this when the request set `logprobs: true`, so that case
    /// does not arise in practice.
    fn build_choice_logprobs(
        tokenizer: &lattice_inference::tokenizer::bpe::BpeTokenizer,
        token_logprobs: &[TokenLogprob],
    ) -> ChoiceLogprobs {
        let content = token_logprobs
            .iter()
            .map(|tl| {
                let (token, bytes) = render_token_logprob(tokenizer, tl.token_id);
                let top_logprobs = tl
                    .top
                    .iter()
                    .map(|alt| {
                        let (token, bytes) = render_token_logprob(tokenizer, alt.token_id);
                        TopLogprobEntry {
                            token,
                            logprob: alt.logprob,
                            bytes,
                        }
                    })
                    .collect();
                TokenLogprobEntry {
                    token,
                    logprob: tl.logprob,
                    bytes,
                    top_logprobs,
                }
            })
            .collect();
        ChoiceLogprobs { content }
    }

    // Handlers
    // -----------------------------------------------------------------------

    pub async fn health() -> Json<HealthResponse> {
        Json(HealthResponse { status: "ok" })
    }

    /// `GET /` (ADR-080 C2): a minimal engine-
    /// identity/endpoint-discovery document, in the same shape
    /// `lattice_serve.rs` already served on its own daemon -- this binary
    /// had no equivalent route at all, an undocumented route-set divergence
    /// between the two binaries -- the routes did not actually match 1:1
    /// until this route landed.
    pub async fn root() -> Json<Value> {
        Json(lattice_inference::serve::root_body())
    }

    /// `GET /v1/models` (ADR-080 C2, #746's sibling gap): advertises the
    /// single loaded model, in the same shape `lattice_serve.rs` already
    /// served on its own daemon -- this binary had no equivalent route at
    /// all before this change.
    pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Json(lattice_inference::serve::models_list_body(
            &state.model_id,
            created,
        ))
    }

    /// Test adapter for the shared normalization cascade without a real
    /// context-window check. Production uses `prepare_chat_request`, which
    /// supplies the prompt-aware check to the same shared function.
    #[cfg(test)]
    fn validate_chat_request(
        req: &ChatCompletionRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<ContractValidatedChatRequest, ApiError> {
        let defaults = GenerationDefaults {
            max_tokens: default_max_tokens,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            reasoning_budget: None,
        };
        let (validated, ()) = normalize_request(
            req,
            defaults,
            ServeProfile::lattice(model_id, max_tokens_cap),
            |_, _| Ok(()),
        )?;
        Ok(validated)
    }

    /// Output of the full pre-generation validation cascade, ready for
    /// `gen_cfg` construction.
    #[derive(Debug)]
    struct PreparedChatRequest {
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        logprobs: Option<usize>,
        prompt: String,
        stop_strings: Vec<String>,
        seed: Option<u64>,
        stream: bool,
    }

    /// Production entry point for the shared `normalize_request` cascade:
    /// supplies the prompt-aware context-window check (rendering the chat
    /// template, tokenizing it, then calling the shared
    /// `validate_context_window`) as the `check_context` closure, in the
    /// exact order the original inline `chat_completions` cascade used:
    /// `stop` is validated *last*, after both the served-model hard
    /// requirements and the context-window check that guards against a
    /// panic in the blocking generation path. A request that is both
    /// over-context and carries a malformed `stop` field must fail with
    /// `context_length_exceeded`, not a stop-parsing error — pinned by
    /// `cm_serve_context_window_checked_before_stop_parsing`.
    ///
    /// `tokenize_len`/`max_context` are threaded through as thunks (rather
    /// than a `&ModelBackend`) so this whole cascade — including the
    /// ordering — is testable without constructing a real model: the
    /// rendered `prompt` that `tokenize_len` needs only exists once
    /// `validate_chat_request` has already run, so the thunk form lets a
    /// test control the token count `check_context_window` sees without
    /// having to fake a tokenizer.
    fn prepare_chat_request(
        req: &ChatCompletionRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        tokenize_len: impl FnOnce(&str) -> usize,
        max_context: impl FnOnce() -> usize,
    ) -> Result<PreparedChatRequest, ApiError> {
        let defaults = GenerationDefaults {
            max_tokens: default_max_tokens,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            reasoning_budget: None,
        };
        let (validated, prompt) = normalize_request(
            req,
            defaults,
            ServeProfile::lattice(model_id, max_tokens_cap),
            |messages, max_tokens| {
                let prompt = format_chat_template(messages);
                let prompt_token_count = tokenize_len(&prompt);
                validate_context_window(prompt_token_count, max_tokens, max_context())?;
                Ok(prompt)
            },
        )?;
        let ContractValidatedChatRequest {
            messages,
            max_tokens,
            temperature,
            top_p,
            logprobs,
            stop_strings,
            seed,
            stream,
            ..
        } = validated;

        Ok(PreparedChatRequest {
            messages,
            max_tokens,
            temperature,
            top_p,
            logprobs,
            prompt,
            stop_strings,
            seed,
            stream,
        })
    }

    /// The `on_token`/`should_cancel` composition for CPU-style streaming
    /// generation, constructed in exactly ONE place and shared by the real
    /// `ModelBackend::Cpu` arm and the test-only `CpuFakeGenerate` arm below
    /// (ADR-080 C2). Before this,
    /// each arm rebuilt `move || *cancel_rx.borrow()` independently, so a
    /// a mutation that broke ONLY the real `Cpu` arm's
    /// predicate left the `CpuFakeGenerate`-only post-drop oracle green --
    /// it was pinning a copy, not the production call site. Both arms now
    /// funnel through this one function, so mutating the shared predicate
    /// here is observed by the fake-arm-driven test too: there is no
    /// separate copy left unmutated.
    #[allow(clippy::type_complexity)]
    fn spawn_cpu_style_streaming_generation(
        tx: futures::channel::mpsc::UnboundedSender<StreamMsg>,
        cancel_rx: tokio::sync::watch::Receiver<bool>,
        prompt: String,
        gen_cfg: lattice_inference::model::qwen35_config::GenerateConfig,
        generate: Arc<
            dyn Fn(
                    &str,
                    &lattice_inference::model::qwen35_config::GenerateConfig,
                    &mut dyn FnMut(&str) -> bool,
                    &mut dyn FnMut() -> bool,
                )
                    -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                + Send
                + Sync,
        >,
        finish_streaming: impl FnOnce(GenerateOutput) + Send + 'static,
    ) {
        tokio::task::spawn_blocking(move || {
            let tx_delta = tx.clone();
            let mut on_token = move |delta: &str| {
                tx_delta
                    .unbounded_send(StreamMsg::Delta(delta.to_string()))
                    .is_ok()
            };
            let mut should_cancel = move || *cancel_rx.borrow();
            let result = generate(&prompt, &gen_cfg, &mut on_token, &mut should_cancel);
            match result {
                Ok(output) => finish_streaming(output),
                Err(e) => {
                    eprintln!("generation error (streaming): {e}");
                    let _ = tx.unbounded_send(StreamMsg::Failed);
                }
            }
        });
    }

    /// Axum route entry point. Takes the raw request body instead of
    /// `Json<ChatCompletionRequest>` so `require_json_content_type` can run
    /// against the raw headers before the body is read (see below). The
    /// message-count bound is enforced inline during the single
    /// `serde_json::from_slice::<ChatCompletionRequest>` parse below
    /// (`serve::contract::deserialize_bounded_messages`), so a sub-body-cap
    /// request built from tens of thousands of tiny messages is rejected
    /// without materializing a `Vec<Message>` entry for each one -- there is
    /// no separate raw-bytes pass over `messages` ahead of that parse.
    ///
    /// `to_bytes(.., REQUEST_BODY_LIMIT_BYTES)` enforces the same cap the
    /// router's `DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES)` layer
    /// already applies to the underlying body stream, so the existing 413
    /// behavior below is unchanged.
    ///
    /// Switching from `Json` to a raw body also dropped `Json`'s own
    /// Content-Type enforcement (a security gap: a body with a valid JSON
    /// payload but `Content-Type: text/plain` -- or no `Content-Type` at
    /// all -- previously got a free 415 from the `Json` extractor before
    /// this handler ever ran). Restored via
    /// [`lattice_inference::serve::require_json_content_type`], checked
    /// against `headers` and the request rejected *before* the body is
    /// read at all: unlike `Result<Bytes, BytesRejection>` (an axum
    /// `FromRequest` extractor that fully buffers the body as part of
    /// argument extraction, before this function body ever runs), taking
    /// the raw `Body` here defers reading the body to the explicit
    /// `to_bytes` call below, so an invalid-MIME request never pays the
    /// buffering cost, mirroring `lattice_serve.rs`'s equivalent handler.
    pub async fn chat_completions(
        State(state): State<AppState>,
        headers: axum::http::HeaderMap,
        body: axum::body::Body,
    ) -> Result<Response, ApiError> {
        lattice_inference::serve::require_json_content_type(&headers)?;

        // Surface a body-length-limit rejection as a structured 413
        // response; any other body-buffering failure (e.g. a client
        // disconnecting mid-stream) is not a size violation and gets the
        // same non-413 invalid-body response as a malformed JSON body.
        let bytes = axum::body::to_bytes(body, REQUEST_BODY_LIMIT_BYTES)
            .await
            .map_err(|err| {
                let is_length_limit = std::error::Error::source(&err)
                    .is_some_and(<dyn std::error::Error>::is::<http_body_util::LengthLimitError>);
                if is_length_limit {
                    return ApiError::PayloadTooLarge {
                        message: "request body exceeds 1 MiB limit".to_string(),
                    };
                }
                eprintln!("invalid request body: {err}");
                ApiError::BadRequest {
                    message: "invalid JSON request body".to_string(),
                    code: "invalid_request_body",
                }
            })?;

        let req: ChatCompletionRequest = serde_json::from_slice(&bytes).map_err(|err| {
            if lattice_inference::serve::contract::is_message_flood_error(&err) {
                return ApiError::BadRequest {
                    message: lattice_inference::serve::contract::message_flood_text(),
                    code: "invalid_request_body",
                };
            }
            eprintln!("invalid request body: {err}");
            ApiError::BadRequest {
                message: "invalid JSON request body".to_string(),
                code: "invalid_request_body",
            }
        })?;

        chat_completions_with_request(State(state), req).await
    }

    /// The full chat-completions cascade, taking an already-parsed request.
    /// Split out of [`chat_completions`] so tests can construct a
    /// `ChatCompletionRequest` directly (a Rust struct literal) without
    /// round-tripping it through JSON bytes -- those tests exercise
    /// generation/streaming behavior, not the raw-bytes preflight, which is
    /// covered separately in `serve::contract`'s own tests and this
    /// binary's router-level tests.
    async fn chat_completions_with_request(
        State(state): State<AppState>,
        req: ChatCompletionRequest,
    ) -> Result<Response, ApiError> {
        let PreparedChatRequest {
            messages: _normalized_messages,
            max_tokens,
            temperature,
            top_p,
            logprobs,
            prompt,
            stop_strings,
            seed,
            stream,
        } = prepare_chat_request(
            &req,
            &state.model_id,
            state.default_max_tokens,
            state.max_tokens_cap,
            |p| state.model.tokenize_len(p),
            || state.model.max_context(),
        )?;

        let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
            max_new_tokens: max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            logprobs,
            ..Default::default()
        };

        // Metal-only: reuse the exact messages normalized alongside the CPU
        // prompt, so role/content validation and allocation happen once.
        #[cfg(feature = "metal-gpu")]
        let chat_messages = _normalized_messages;
        // CPU-only builds never render `_normalized_messages` (the CPU
        // closures below capture only `cpu_model`/`prompt`/`gen_cfg`) -- drop
        // it here instead of letting it ride, unused, across the
        // `spawn_blocking(...).await` that follows.
        #[cfg(not(feature = "metal-gpu"))]
        drop(_normalized_messages);

        let model = state.model.clone();

        // Compute shared response metadata before branching on stream flag.
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let seq = state.request_counter.fetch_add(1, Ordering::Relaxed);
        let response_id = format!("chatcmpl-{created}-{seq}");

        if stream {
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
            // string.
            //
            // Disconnect cancellation (ADR-080 C2, #744): `cancel_guard` is
            // moved into the `body_stream` closure below, so it drops the
            // instant axum drops this response's stream (client disconnect).
            // Dropping it flips `cancel_rx` to `true`, which both backends
            // poll independently of `on_token` (before prefill, immediately
            // after prefill, and at the top of every decode iteration) —
            // closing the gap the old comment here used to document as "a
            // future refinement".
            let (tx, rx) = futures::channel::mpsc::unbounded::<StreamMsg>();
            let (cancel_guard, cancel_rx) = lattice_inference::serve::cancel_pair();

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
                    // Delegates through the SAME shared helper the test-only
                    // `CpuFakeGenerate` arm below uses -- only the generation call itself
                    // (`cpu_model.generate_streaming_with_cancel` here, an
                    // injected closure there) differs; the `should_cancel`
                    // predicate is constructed once, by the helper, for both.
                    #[allow(clippy::type_complexity)]
                    let generate: Arc<
                        dyn Fn(
                                &str,
                                &lattice_inference::model::qwen35_config::GenerateConfig,
                                &mut dyn FnMut(&str) -> bool,
                                &mut dyn FnMut() -> bool,
                            )
                                -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                            + Send
                            + Sync,
                    > = Arc::new(move |p, c, on_token, should_cancel| {
                        cpu_model.generate_streaming_with_cancel(p, c, on_token, should_cancel)
                    });
                    spawn_cpu_style_streaming_generation(
                        tx,
                        cancel_rx,
                        prompt,
                        gen_cfg,
                        generate,
                        finish_streaming,
                    );
                }
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { handle, .. } => {
                    // #939: submit synchronously, before any SSE response is
                    // built. `handle.submit` runs `MetalWorkerClient::submit`'s
                    // admission check (#932) directly on this task -- not
                    // inside the `tokio::spawn` below -- so an admission
                    // rejection propagates via `?` as an ordinary
                    // `ApiError::ServiceUnavailable` (503) before this
                    // handler ever returns `Sse::new(...)`. Only draining the
                    // already-admitted job happens in the detached task.
                    let mut rx = handle.submit(chat_messages, gen_cfg, cancel_rx)?;
                    tokio::spawn(async move {
                        let tx_delta = tx.clone();
                        let result = MetalHandle::drain(&mut rx, move |delta| {
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
                // ADR-080 C2: goes
                // through the exact same `spawn_cpu_style_streaming_generation`
                // helper as the real `Cpu` arm above -- there is no separate
                // `should_cancel` construction here to leave un-mutated. Only
                // the injected `generate` closure differs; this is the seam a
                // post-drop cancellation probe test substitutes.
                #[cfg(all(feature = "test-utils", test))]
                ModelBackend::CpuFakeGenerate { generate, .. } => {
                    spawn_cpu_style_streaming_generation(
                        tx,
                        cancel_rx,
                        prompt,
                        gen_cfg,
                        generate,
                        finish_streaming,
                    );
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
                // Keeps `cancel_guard` alive for exactly as long as this
                // stream is: the moment axum drops the whole SSE response
                // (client disconnect), this closure -- and the guard moved
                // into it -- drops too, flipping `cancel_rx` to `true`.
                let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;
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
                        // The producer already logged the specific cause; keep
                        // client-visible detail generic while making failure
                        // distinguishable from a genuine stop condition.
                        let data = serde_json::json!({
                            "error": {
                                "message": "inference failed",
                                "type": "server_error",
                                "code": "internal_error",
                                "param": null,
                            }
                        })
                        .to_string();
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
                    .generate_streaming(chat_messages, gen_cfg, |_delta| true)
                    .await
                    .map_err(|e| match e {
                        // #939: admission rejection (#932's outstanding-job
                        // cap) must surface as the shared 503 `server_busy`
                        // envelope unchanged -- collapsing it into
                        // `ApiError::Internal` here made every non-streaming
                        // Metal admission rejection an indistinguishable 500.
                        ApiError::ServiceUnavailable { message } => {
                            ApiError::ServiceUnavailable { message }
                        }
                        other => {
                            eprintln!("generation error (metal): {other:?}");
                            ApiError::Internal {
                                message: "inference failed".to_string(),
                            }
                        }
                    })?,
                // ADR-080 C2 added
                // this variant for the streaming arm's cancellation probe
                // only, so non-streaming used to bypass the injected
                // `generate` closure entirely and delegate straight to the
                // real tiny model. Issue #828's field-level parity rows need
                // a NON-streaming seam too (deterministic content/usage
                // counts for `FieldExpectation::Eq` checks), so this now
                // goes through the exact same injected closure the
                // streaming arm uses -- `on_token`/`should_cancel` are
                // no-ops here (this arm is never the cancellation probe's
                // concern), matching how `model.generate()` itself has no
                // early-stop/cancel hooks either.
                #[cfg(all(feature = "test-utils", test))]
                ModelBackend::CpuFakeGenerate { generate, .. } => {
                    tokio::task::spawn_blocking(move || {
                        generate(&prompt, &gen_cfg, &mut |_delta: &str| true, &mut || false)
                    })
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

            // #585: only render logprobs (and touch the tokenizer for it) when
            // the request actually asked for them — `logprobs` is `None` on
            // every other request, so this is a no-op on the default path.
            let choice_logprobs = logprobs
                .is_some()
                .then(|| build_choice_logprobs(state.model.tokenizer(), &output.token_logprobs));

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
                    logprobs: choice_logprobs,
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
            .route("/", get(root))
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
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
        use axum::http::StatusCode;
        use lattice_inference::forward::metal_qwen35::ChatRole;

        /// #832 checklist: "Add a Metal-feature test asserting an explicit
        /// common-worker marker, so a silently-reintroduced per-binary
        /// fallback cannot pass green." `MetalHandle::client`'s field type
        /// is private, so this test can only be written from inside
        /// `mod serve` (same module, so private-field access is allowed) —
        /// it constructs a `MetalHandle` directly from a
        /// `lattice_inference::serve::metal_worker::MetalWorkerClient`,
        /// which only type-checks if `MetalHandle` still wraps that exact
        /// shared type. A private per-binary fallback worker (any type
        /// other than the shared `MetalWorkerClient`) would fail to compile
        /// here, not just fail some behavioral assertion at runtime.
        /// `test-utils`-gated: `test_client_and_jobs` (the only way to
        /// obtain a real `MetalWorkerClient` without a live GPU worker
        /// thread) lives behind that feature — see its own doc comment.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        #[test]
        fn metal_handle_is_backed_by_the_shared_metal_worker_client() {
            fn build_from_shared_client(
                client: lattice_inference::serve::metal_worker::MetalWorkerClient,
            ) -> MetalHandle {
                MetalHandle { client }
            }
            let (client, _jobs_rx) = lattice_inference::serve::metal_worker::test_client_and_jobs();
            let _handle: MetalHandle = build_from_shared_client(client);
        }

        /// #939 regression coverage: the unified `lattice serve` HTTP
        /// adapter's admission ordering. Before this fix, the Metal
        /// streaming arm called `MetalWorkerClient::submit` (the ONE way
        /// admission (#932) can reject a request) INSIDE a detached
        /// `tokio::spawn` task, after `chat_completions` had already
        /// returned `Sse::new(...).into_response()` -- so an admission
        /// rejection surfaced as a normal-looking stream that immediately
        /// ended, never as HTTP 503. The non-streaming arm separately
        /// collapsed `ApiError::ServiceUnavailable` into a generic 500. Both
        /// arms are driven here through the REAL `router()` + a real
        /// worker thread (`spawn_fake_with_cap`, cap=1, `generate` blocked
        /// on a channel so exactly one request is provably "in flight"),
        /// mirroring `lattice_serve.rs`'s own
        /// `real_router_admission_cap::chat_completions_returns_503_json_envelope_when_admission_cap_reached`
        /// -- that test alone did not catch this bug because it only
        /// covers the daemon's separate, already-correct adapter.
        #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
        mod admission_cap_939 {
            use super::*;
            use axum::body::Body;
            use axum::http::StatusCode;
            use lattice_inference::serve::metal_worker::{
                ContextWindowPolicy, spawn_fake_with_cap,
            };
            use std::sync::mpsc as std_mpsc;
            use tower::ServiceExt as _;

            /// A real (if tiny) tokenizer -- NOT a hand-rolled minimal
            /// vocab. `lattice_serve.rs`'s equivalent
            /// `real_router_admission_cap::single_slot_blocking_state` uses
            /// the same `test_support::tiny_zero_model` tokenizer for
            /// exactly this reason: a hand-rolled single-entry vocab (e.g.
            /// just `{"hi": 0}`) lacks the byte-level fallback tokens a
            /// real BPE tokenizer's byte-encoder table relies on, so
            /// tokenizing the full rendered ChatML prompt against it can
            /// silently misbehave in ways a real tokenizer never does.
            fn tiny_tokenizer() -> lattice_inference::tokenizer::bpe::BpeTokenizer {
                lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone()
            }

            /// A real worker thread (cap=1) whose `generate` blocks on
            /// `unblock_rx.recv()` until the test releases it, so exactly
            /// one request can be "in flight" for as long as the test
            /// needs -- long enough to prove a second concurrent request is
            /// rejected by admission rather than racing to observe an
            /// already-freed slot. Mirrors `lattice_serve.rs`'s
            /// `real_router_admission_cap::single_slot_blocking_state`.
            fn single_slot_blocking_state()
            -> (AppState, std_mpsc::Sender<()>, std_mpsc::Receiver<()>) {
                let (unblock_tx, unblock_rx) = std_mpsc::channel::<()>();
                let unblock_rx = std::sync::Mutex::new(unblock_rx);
                let (started_tx, started_rx) = std_mpsc::channel::<()>();
                let client = spawn_fake_with_cap(
                    1,
                    ContextWindowPolicy::PromptAndMaxTokens,
                    4096,
                    tiny_tokenizer(),
                    move |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
                        let _ = started_tx.send(());
                        // Blocks the dedicated fake-worker OS thread (not
                        // the tokio runtime) until the test explicitly lets
                        // it proceed.
                        let _ = unblock_rx.lock().unwrap().recv();
                        Ok(GenerateOutput {
                            text: String::new(),
                            token_ids: vec![],
                            prompt_tokens,
                            generated_tokens: 0,
                            stopped: true,
                            stop_reason: None,
                            token_logprobs: vec![],
                        })
                    },
                );
                let state = AppState {
                    model: ModelBackend::Metal {
                        handle: MetalHandle { client },
                        tokenizer: Arc::new(tiny_tokenizer()),
                        max_context: 4096,
                    },
                    default_max_tokens: 16,
                    max_tokens_cap: 4096,
                    model_id: "test-model".to_string(),
                    request_counter: Arc::new(AtomicU64::new(0)),
                };
                (state, unblock_tx, started_rx)
            }

            fn chat_request(stream: bool) -> axum::http::Request<Body> {
                let body = if stream {
                    r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#
                } else {
                    r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
                };
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("fixture request must build")
            }

            /// Asserts the shared 503 `server_busy` JSON envelope, AND that
            /// no SSE response was ever committed (issue #939's exact
            /// regression for the streaming arm: a 200 SSE response whose
            /// body happened to end immediately would previously slip past
            /// a status-code-only assertion).
            async fn assert_503_server_busy_no_sse(
                response: axum::http::Response<Body>,
                case: &str,
            ) {
                assert_eq!(
                    response.status(),
                    StatusCode::SERVICE_UNAVAILABLE,
                    "{case}: a request submitted while the admission cap is full must be \
                     rejected with HTTP 503, fail-fast, before any response -- streaming \
                     or not -- is committed"
                );
                let content_type = response
                    .headers()
                    .get(axum::http::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                assert!(
                    !content_type.contains("text/event-stream"),
                    "{case}: a 503 rejection must never carry an SSE content-type -- that \
                     would mean a streaming response had already committed before \
                     admission was checked: {content_type}"
                );
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("503 response body must be readable");
                let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                    panic!("{case}: 503 response must be JSON, not an SSE body: {e}")
                });
                assert_eq!(value["error"]["code"], "server_busy", "{case}: {value}");
                assert_eq!(value["error"]["type"], "server_error", "{case}: {value}");
                assert!(value["error"]["param"].is_null(), "{case}: {value}");
                assert!(
                    !value["error"]["message"]
                        .as_str()
                        .unwrap_or_default()
                        .is_empty(),
                    "{case}: 503 envelope must carry a human-readable message: {value}"
                );
            }

            #[tokio::test]
            async fn chat_completions_streaming_returns_503_before_sse_commit_at_cap() {
                let (state, unblock_tx, started_rx) = single_slot_blocking_state();
                let app = router(state);

                let app1 = app.clone();
                let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
                tokio::task::spawn_blocking(move || started_rx.recv())
                    .await
                    .expect("blocking wait must not panic")
                    .expect("request 1's worker-thread generate() must signal it started");

                let response2 = app
                    .clone()
                    .oneshot(chat_request(true))
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_503_server_busy_no_sse(response2, "stream:true").await;

                unblock_tx.send(()).expect("unblock send must succeed");
                let response1 = handle1
                    .await
                    .expect("request 1's task must not panic")
                    .expect("router must produce a response, not a transport error");
                assert_eq!(
                    response1.status(),
                    StatusCode::OK,
                    "request 1 itself was never over any cap and must succeed normally"
                );
            }

            #[tokio::test]
            async fn chat_completions_non_streaming_returns_503_server_busy_not_500_at_cap() {
                let (state, unblock_tx, started_rx) = single_slot_blocking_state();
                let app = router(state);

                let app1 = app.clone();
                let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
                tokio::task::spawn_blocking(move || started_rx.recv())
                    .await
                    .expect("blocking wait must not panic")
                    .expect("request 1's worker-thread generate() must signal it started");

                let response2 = app
                    .clone()
                    .oneshot(chat_request(false))
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_503_server_busy_no_sse(response2, "non-streaming").await;

                unblock_tx.send(()).expect("unblock send must succeed");
                let response1 = handle1
                    .await
                    .expect("request 1's task must not panic")
                    .expect("router must produce a response, not a transport error");
                assert_eq!(
                    response1.status(),
                    StatusCode::OK,
                    "request 1 itself was never over any cap and must succeed normally"
                );
            }
        }

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
        fn chat_template_multi_message_chatml() {
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
            let prompt = format_chat_template(&to_chat_messages(&messages).unwrap());
            assert!(prompt.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
            assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
        }

        // Role/content-part rejection cases are covered by the
        // `to_chat_messages_rejects_*` tests below -- `to_chat_messages` is
        // the sole validation entry point the ChatML renderer sits behind
        // (#668), so there is no second renderer left to test separately.

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
                token_logprobs: vec![],
            };
            assert_eq!(super::finish_reason_for(&cap), "length");

            let natural = GenerateOutput {
                text: "hello".into(),
                token_ids: vec![1, 2, 3],
                prompt_tokens: 10,
                generated_tokens: 3,
                stopped: true,
                stop_reason: Some(lattice_inference::StopReason::Eos),
                token_logprobs: vec![],
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
                token_logprobs: vec![],
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
                token_logprobs: vec![],
            };
            assert_eq!(super::finish_reason_for(&output), "length");
        }

        #[test]
        fn reject_unsupported_stream_true_ok() {
            // stream=true is now handled by the streaming path and must NOT be
            // rejected by reject_unsupported.
            let req = ChatCompletionRequest {
                model: Some("m".to_string()),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
                reasoning_budget: None,
                stream: Some(true),
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                top_logprobs: None,
                n: None,
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        #[test]
        fn reject_unsupported_stream_and_logprobs_rejected() {
            // #585: logprobs is implemented on the non-streaming path only;
            // combined with stream: true it must be rejected, not silently
            // ignored.
            let req = ChatCompletionRequest {
                stream: Some(true),
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
                model: Some("m".to_string()),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
                reasoning_budget: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                top_logprobs: None,
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
                model: Some("m".to_string()),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
                reasoning_budget: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: Some(ResponseFormat {
                    r#type: "json_object".to_string(),
                    json_schema: None,
                }),
                tools: None,
                tool_choice: None,
                logprobs: None,
                top_logprobs: None,
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
                model: Some("m".to_string()),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
                reasoning_budget: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                top_logprobs: None,
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
        fn reject_unsupported_logprobs_true_ok() {
            // #585: logprobs is now implemented on the non-streaming path, so a
            // standalone `logprobs: true` (no `stream: true`) must be accepted
            // here — validation of the value itself is `validate_logprobs`'s job.
            let req = ChatCompletionRequest {
                logprobs: Some(true),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
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
                    json_schema: None,
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
        // format_chat_template (via to_chat_messages) — additional rendering cases
        // -----------------------------------------------------------------------

        #[test]
        fn chat_template_user_only() {
            let msgs = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
            }];
            let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
            assert_eq!(
                prompt,
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
            );
        }

        #[test]
        fn chat_template_multi_turn_assistant() {
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
            let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
            assert!(prompt.contains("<|im_start|>user\nq1<|im_end|>"));
            assert!(prompt.contains("<|im_start|>assistant\na1<|im_end|>"));
            assert!(prompt.contains("<|im_start|>user\nq2<|im_end|>"));
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
        }

        #[test]
        fn chat_template_content_parts_text_ok() {
            let msgs = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "hello".to_string(),
                    },
                    ContentPart::Text {
                        text: " world".to_string(),
                    },
                ]),
            }];
            let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
            assert!(prompt.contains("<|im_start|>user\nhello world<|im_end|>"));
        }

        // -----------------------------------------------------------------------
        // `to_chat_messages` validation (#661, CPU-available since #668) — the
        // sole validation entry point the ChatML renderer sits behind.
        // -----------------------------------------------------------------------

        #[test]
        fn to_chat_messages_rejects_invalid_role() {
            let messages = vec![Message {
                role: "function".to_string(),
                content: MessageContent::Text("data".to_string()),
            }];
            let err = to_chat_messages(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_role",
                    ..
                }
            ));
        }

        #[test]
        fn to_chat_messages_rejects_tool_role() {
            let messages = vec![Message {
                role: "tool".to_string(),
                content: MessageContent::Text("result".to_string()),
            }];
            let err = to_chat_messages(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn to_chat_messages_rejects_developer_role() {
            let messages = vec![Message {
                role: "developer".to_string(),
                content: MessageContent::Text("system prompt".to_string()),
            }];
            let err = to_chat_messages(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn to_chat_messages_rejects_non_text_content_part() {
            let messages = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: lattice_inference::serve::contract::ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                }]),
            }];
            let err = to_chat_messages(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn to_chat_messages_accepts_valid_roles() {
            let messages = vec![
                Message {
                    role: "system".to_string(),
                    content: MessageContent::Text("Be helpful.".to_string()),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("q1".to_string()),
                },
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text("a1".to_string()),
                },
            ];
            let chat_messages = to_chat_messages(&messages).unwrap();
            assert_eq!(chat_messages.len(), 3);
            assert_eq!(chat_messages[0].role, ChatRole::System);
            assert_eq!(chat_messages[0].content, "Be helpful.");
            assert_eq!(chat_messages[1].role, ChatRole::User);
            assert_eq!(chat_messages[2].role, ChatRole::Assistant);
        }

        // -----------------------------------------------------------------------
        // Error envelope JSON shape
        // -----------------------------------------------------------------------

        /// Drains an `axum::response::Response` body into a parsed `Value`,
        /// for asserting on the shared `lattice_inference::serve::ApiError`
        /// envelope shape (ADR-080 C2, #782) from this binary's own tests.
        async fn response_json(response: axum::response::Response) -> serde_json::Value {
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body reads");
            serde_json::from_slice(&body).expect("response body is valid JSON")
        }

        #[tokio::test]
        async fn error_envelope_bad_request_shape() {
            let err = ApiError::BadRequest {
                message: "test error".to_string(),
                code: "invalid_request",
            };
            // Variant check kept separate so we know err itself was constructed correctly.
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_request",
                    ..
                }
            ));
            // Verify the shared ApiError serialises to the OpenAI envelope shape:
            // {"error":{"message":"...","type":"invalid_request_error","code":"...","param":null}}
            let response = ApiError::BadRequest {
                message: "test error".to_string(),
                code: "invalid_request",
            }
            .into_response();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let v = response_json(response).await;
            assert!(v["error"].is_object(), "top-level key must be 'error'");
            assert_eq!(v["error"]["message"], "test error");
            assert_eq!(v["error"]["type"], "invalid_request_error");
            assert_eq!(v["error"]["code"], "invalid_request");
            assert!(v["error"]["param"].is_null());
        }

        #[tokio::test]
        async fn error_envelope_payload_too_large_shape() {
            let response = ApiError::PayloadTooLarge {
                message: "request body exceeds 1 MiB limit".to_string(),
            }
            .into_response();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
            let v = response_json(response).await;
            assert_eq!(v["error"]["code"], "request_body_too_large");
        }

        #[tokio::test]
        async fn error_envelope_internal_shape() {
            let response = ApiError::Internal {
                message: "inference failed".to_string(),
            }
            .into_response();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            let v = response_json(response).await;
            assert_eq!(v["error"]["type"], "server_error");
            assert_eq!(v["error"]["code"], "internal_error");
        }

        // -----------------------------------------------------------------------
        // message content normalization (via the shared normalize_messages,
        // exercised here through the `to_chat_messages` alias)
        // -----------------------------------------------------------------------

        #[test]
        fn message_text_plain_string() {
            let messages = [Message {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
            }];
            assert_eq!(to_chat_messages(&messages).unwrap()[0].content, "hello");
        }

        #[test]
        fn message_text_parts_concatenates() {
            let messages = [Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "foo".to_string(),
                    },
                    ContentPart::Text {
                        text: "bar".to_string(),
                    },
                ]),
            }];
            assert_eq!(to_chat_messages(&messages).unwrap()[0].content, "foobar");
        }

        #[test]
        fn message_text_parts_rejects_image() {
            let messages = [Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: lattice_inference::serve::contract::ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                }]),
            }];
            let err = to_chat_messages(&messages).unwrap_err();
            match err {
                ApiError::BadRequest { message, code } => {
                    assert_eq!(code, "unsupported_feature");
                    assert_eq!(message, "image input requires a vision-capable model");
                }
                other => panic!("expected BadRequest, got {other:?}"),
            }
        }

        #[test]
        fn message_text_parts_rejects_unknown_part_type() {
            let messages = [Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![ContentPart::Unsupported {
                    kind: "file".to_string(),
                }]),
            }];
            let err = to_chat_messages(&messages).unwrap_err();
            match err {
                ApiError::BadRequest { message, code } => {
                    assert_eq!(code, "unsupported_feature");
                    assert_eq!(
                        message,
                        "content part type 'file' is not supported; only 'text' parts are accepted"
                    );
                }
                other => panic!("expected BadRequest, got {other:?}"),
            }
        }

        // -----------------------------------------------------------------------
        // validate_logprobs (#585)
        // -----------------------------------------------------------------------

        #[test]
        fn validate_logprobs_absent_disables_capture() {
            assert_eq!(validate_logprobs(None, None).unwrap(), None);
        }

        #[test]
        fn validate_logprobs_false_disables_capture() {
            assert_eq!(validate_logprobs(Some(false), None).unwrap(), None);
        }

        #[test]
        fn validate_logprobs_true_no_top_logprobs_defaults_to_zero() {
            // logprobs: true with no top_logprobs still captures the sampled
            // token's own logprob, just with no alternatives.
            assert_eq!(validate_logprobs(Some(true), None).unwrap(), Some(0));
        }

        #[test]
        fn validate_logprobs_true_with_top_logprobs_ok() {
            assert_eq!(validate_logprobs(Some(true), Some(5)).unwrap(), Some(5));
        }

        #[test]
        fn validate_logprobs_top_logprobs_at_boundary_twenty_ok() {
            assert_eq!(validate_logprobs(Some(true), Some(20)).unwrap(), Some(20));
        }

        #[test]
        fn validate_logprobs_top_logprobs_over_twenty_rejected() {
            let err = validate_logprobs(Some(true), Some(21)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_top_logprobs",
                    ..
                }
            ));
        }

        #[test]
        fn validate_logprobs_top_logprobs_without_logprobs_true_rejected() {
            let err = validate_logprobs(None, Some(5)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_request",
                    ..
                }
            ));
        }

        #[test]
        fn validate_logprobs_top_logprobs_with_logprobs_false_rejected() {
            let err = validate_logprobs(Some(false), Some(5)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_request",
                    ..
                }
            ));
        }

        // -----------------------------------------------------------------------
        // render_token_logprob / build_choice_logprobs (#585)
        // -----------------------------------------------------------------------

        /// Tiny in-memory tokenizer for logprob-rendering tests — no merges,
        /// just a fixed id -> token vocabulary large enough to exercise both a
        /// known and an unresolved token id. "Hello"/"world" are plain ASCII in
        /// the printable range the GPT-2 byte table maps to itself, so they
        /// round-trip byte-for-byte through `byte_decode_token[_bytes]`.
        fn logprob_test_tokenizer() -> lattice_inference::tokenizer::bpe::BpeTokenizer {
            let vocab: std::collections::HashMap<String, u32> =
                [("Hello".to_string(), 0u32), ("world".to_string(), 1u32)]
                    .into_iter()
                    .collect();
            lattice_inference::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, vec![])
                .expect("in-memory test vocab must construct")
        }

        #[test]
        fn render_token_logprob_resolves_known_token() {
            let tokenizer = logprob_test_tokenizer();
            let (token, bytes) = render_token_logprob(&tokenizer, 0);
            assert_eq!(token, "Hello");
            assert_eq!(bytes, Some(b"Hello".to_vec()));
        }

        #[test]
        fn render_token_logprob_unresolved_id_fails_closed() {
            // Token id 999 does not exist in the 2-entry test vocab: this must
            // fail closed with a visibly synthetic token and no bytes, never panic.
            let tokenizer = logprob_test_tokenizer();
            let (token, bytes) = render_token_logprob(&tokenizer, 999);
            assert_eq!(token, "<|unresolved_token_999|>");
            assert_eq!(bytes, None);
        }

        #[test]
        fn build_choice_logprobs_shapes_content_and_alternatives() {
            let tokenizer = logprob_test_tokenizer();
            let token_logprobs = vec![
                TokenLogprob {
                    token_id: 0,
                    logprob: -0.1,
                    top: vec![
                        lattice_inference::model::qwen35_config::TopLogprob {
                            token_id: 0,
                            logprob: -0.1,
                        },
                        lattice_inference::model::qwen35_config::TopLogprob {
                            token_id: 1,
                            logprob: -2.3,
                        },
                    ],
                },
                TokenLogprob {
                    token_id: 1,
                    logprob: -0.05,
                    top: vec![],
                },
            ];
            let choice_logprobs = build_choice_logprobs(&tokenizer, &token_logprobs);
            assert_eq!(choice_logprobs.content.len(), 2);

            assert_eq!(choice_logprobs.content[0].token, "Hello");
            assert_eq!(choice_logprobs.content[0].logprob, -0.1);
            assert_eq!(choice_logprobs.content[0].top_logprobs.len(), 2);
            assert_eq!(choice_logprobs.content[0].top_logprobs[0].token, "Hello");
            assert_eq!(choice_logprobs.content[0].top_logprobs[1].token, "world");

            assert_eq!(choice_logprobs.content[1].token, "world");
            assert_eq!(choice_logprobs.content[1].logprob, -0.05);
            assert!(choice_logprobs.content[1].top_logprobs.is_empty());
        }

        // -----------------------------------------------------------------------
        // Choice.logprobs — JSON shape (#585)
        // -----------------------------------------------------------------------

        #[test]
        fn choice_logprobs_omitted_from_json_when_none() {
            // The no-logprobs-requested response must be byte-identical to
            // before this feature existed: the key is absent, not `null`.
            let choice = Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: "hi".to_string(),
                },
                finish_reason: "stop".to_string(),
                logprobs: None,
            };
            let json = serde_json::to_string(&choice).unwrap();
            assert!(
                !json.contains("logprobs"),
                "logprobs key must be entirely absent when None, got: {json}"
            );
        }

        #[test]
        fn choice_logprobs_present_when_requested() {
            let choice = Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: "hi".to_string(),
                },
                finish_reason: "stop".to_string(),
                logprobs: Some(ChoiceLogprobs {
                    content: vec![TokenLogprobEntry {
                        token: "hi".to_string(),
                        logprob: -0.2,
                        bytes: Some(b"hi".to_vec()),
                        top_logprobs: vec![],
                    }],
                }),
            };
            let json = serde_json::to_string(&choice).unwrap();
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["logprobs"]["content"][0]["token"], "hi");
            assert_eq!(v["logprobs"]["content"][0]["logprob"], -0.2);
            assert_eq!(
                v["logprobs"]["content"][0]["bytes"],
                serde_json::json!([104, 105])
            );
            assert_eq!(
                v["logprobs"]["content"][0]["top_logprobs"],
                serde_json::json!([])
            );
        }

        // -----------------------------------------------------------------------
        // Capability-matrix fixtures (#654) — `validate_chat_request` cascade.
        //
        // Each `#[test]` fn name below is a fixture ID cited from
        // `docs/capability-matrix.md`'s Fixture column; `scripts/check-capability-
        // matrix.sh` greps this file for `fn <fixture_id>` and fails the build if
        // a matrix row cites an ID that no longer exists here. These three checks
        // (model-id match, empty messages, last-role-must-be-user) previously ran
        // only inline in `chat_completions` with no dedicated test at all.
        // -----------------------------------------------------------------------

        fn user_msg(text: &str) -> Message {
            Message {
                role: "user".to_string(),
                content: MessageContent::Text(text.to_string()),
            }
        }

        #[test]
        fn cm_serve_model_mismatch_rejected() {
            let req = ChatCompletionRequest {
                model: Some("some-other-model".to_string()),
                messages: vec![user_msg("hi")],
                ..bare_req()
            };
            let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "model_not_found",
                    ..
                }
            ));
        }

        #[test]
        fn cm_serve_model_match_passes_model_check() {
            let req = ChatCompletionRequest {
                model: Some("served-model".to_string()),
                messages: vec![user_msg("hi")],
                ..bare_req()
            };
            assert!(validate_chat_request(&req, "served-model", 256, 4096).is_ok());
        }

        #[test]
        fn cm_serve_empty_messages_rejected() {
            let req = ChatCompletionRequest {
                model: Some("served-model".to_string()),
                messages: vec![],
                ..bare_req()
            };
            let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_messages",
                    ..
                }
            ));
        }

        #[test]
        fn cm_serve_last_message_not_user_rejected() {
            let req = ChatCompletionRequest {
                model: Some("served-model".to_string()),
                messages: vec![
                    user_msg("hi"),
                    Message {
                        role: "assistant".to_string(),
                        content: MessageContent::Text("hello".to_string()),
                    },
                ],
                ..bare_req()
            };
            let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_messages",
                    ..
                }
            ));
        }

        #[test]
        fn cm_serve_unsupported_feature_rejected_before_model_check() {
            // `reject_unsupported` (tools/n/response_format/stream+logprobs) runs
            // first in the cascade: a request that both targets the wrong model
            // AND asks for `tools` must fail on the tools rejection, not the
            // model-mismatch check, so callers get the more specific error.
            let req = ChatCompletionRequest {
                model: Some("some-other-model".to_string()),
                messages: vec![user_msg("hi")],
                tools: Some(serde_json::json!([{"type": "function"}])),
                ..bare_req()
            };
            let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn cm_serve_stop_sequences_accepted_end_to_end() {
            // Full-cascade check that a well-formed request carrying `stop`
            // resolves through to `PreparedChatRequest.stop_strings` — the
            // capability matrix's "supported" claim for `stop` on this surface.
            let req = ChatCompletionRequest {
                model: Some("served-model".to_string()),
                messages: vec![user_msg("hi")],
                stop: Some(serde_json::json!(["\n\n"])),
                ..bare_req()
            };
            let prepared =
                prepare_chat_request(&req, "served-model", 256, 4096, |_| 1, || 4096).unwrap();
            assert_eq!(prepared.stop_strings, vec!["\n\n".to_string()]);
        }

        #[test]
        fn cm_serve_context_window_checked_before_stop_parsing() {
            // Regression fixture for a refactor bug: extracting stop-sequence
            // parsing into the pre-model validation cascade moved it ahead of
            // the context-window check. The pre-refactor inline sequence
            // checked the context window BEFORE parsing `stop`. A request that is both
            // over-context and carries a malformed `stop` field must
            // therefore fail with `context_length_exceeded`, not a
            // stop-parsing error.
            //
            // This drives `prepare_chat_request` itself (not just its
            // sub-functions in isolation), with a `tokenize_len` thunk that
            // reports the whole context window as already consumed by the
            // prompt — so it is sensitive to a future reordering of the
            // `check_context_window` / `parse_stop_strings` calls inside
            // `prepare_chat_request`, not just to whether each sub-function
            // works in isolation.
            let req = ChatCompletionRequest {
                model: Some("served-model".to_string()),
                messages: vec![user_msg("hi")],
                stop: Some(serde_json::json!([])), // malformed: empty array is rejected
                ..bare_req()
            };
            let err = prepare_chat_request(&req, "served-model", 256, 4096, |_| 4096, || 4096)
                .unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "context_length_exceeded",
                    ..
                }
            ));
        }

        #[test]
        fn cm_serve_logprobs_resolved_end_to_end() {
            // Full-cascade check backing the matrix's "supported, non-streaming
            // only" `logprobs`/`top_logprobs` claim for `lattice serve`.
            let req = ChatCompletionRequest {
                model: Some("served-model".to_string()),
                messages: vec![user_msg("hi")],
                logprobs: Some(true),
                top_logprobs: Some(3),
                ..bare_req()
            };
            let validated = validate_chat_request(&req, "served-model", 256, 4096).unwrap();
            assert_eq!(validated.logprobs, Some(3));
        }

        // -----------------------------------------------------------------------
        // Shared `AppState` builder for the router-level test modules below --
        // both need a real (tiny, deterministic) CPU model, gated behind
        // `test-utils` (see `lattice_inference::model::qwen35::test_support`)
        // for the same reason: bin targets can't see this crate's own
        // `#[cfg(test)]`-only fixtures across the bin/lib compilation
        // boundary, only a real Cargo feature crosses it.
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        fn tiny_state(max_tokens_cap: usize) -> AppState {
            let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
            AppState {
                model: ModelBackend::Cpu(Arc::new(model)),
                default_max_tokens: max_tokens_cap,
                max_tokens_cap,
                model_id: "test-model".to_string(),
                request_counter: Arc::new(AtomicU64::new(0)),
            }
        }

        // -----------------------------------------------------------------------
        // HTTP-level client-disconnect cancellation (ADR-080 C2) -- gated behind `test-utils` (see
        // `lattice_inference::model::qwen35::test_support`) because it needs a
        // real, tiny, deterministic CPU model to exercise the actual
        // `chat_completions` -> `body_stream`'s `cancel_guard` capture ->
        // `generate_streaming_with_cancel` composition end to end, not just the
        // primitive (already unit/mutation-tested in
        // `model/qwen35/generation.rs`) or the guard type (already unit-tested
        // in `serve/mod.rs`) in isolation. "The disclosed
        // HTTP-level disconnect test gap ... does not waive that gate."
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        mod disconnect_cancellation {
            use super::*;
            use http_body_util::BodyExt;
            use std::time::Duration;

            /// The tiny test model's context window is a fixed 1024 tokens
            /// (`test_support::tiny_zero_model`'s `max_position_embeddings`),
            /// so `max_tokens` here must leave room for the rendered prompt
            /// (well under 1024) -- "effectively unbounded relative to this
            /// test's timeouts" rather than the vastly larger figure a real
            /// model's context window would allow.
            const NEAR_MAX_CONTEXT_TOKENS: usize = 900;

            fn tiny_state() -> AppState {
                super::tiny_state(NEAR_MAX_CONTEXT_TOKENS)
            }

            /// Proves the real `chat_completions` streaming composition --
            /// not just its primitives in isolation -- actually stops
            /// generation on client disconnect. A tiny all-zero-weight model
            /// with `max_tokens` near the tiny model's 1024-token context window means
            /// uncancelled generation would keep running far longer than
            /// this test's timeouts; it reads two real SSE frames (proving
            /// generation is genuinely under way) before dropping the
            /// response body to simulate a disconnect.
            ///
            /// Mutation-sensitive to a known regression: if
            /// `let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;`
            /// is removed from `body_stream`'s `flat_map` closure in
            /// `chat_completions`, `cancel_guard` is no longer captured by
            /// anything and drops at the end of `chat_completions`'s own
            /// function body -- i.e. before the response is even returned to
            /// the caller, let alone before the client could disconnect.
            /// That flips `cancel_rx` immediately, so
            /// `generate_streaming_with_cancel`'s pre-prefill checkpoint
            /// returns `Interrupt` with zero tokens generated. Frame 1 (the
            /// role chunk) still arrives unconditionally either way, and a
            /// *second* frame still arrives either way too -- but under the
            /// mutation it is the finish chunk (`finish_reason: "length"` --
            /// the shared `finish_reason()` mapping has no `"interrupt"`
            /// string; a cancelled/interrupted result reports `"length"`
            /// exactly like a token-cap stop, see its own doc comment --
            /// no `content`) sent by `finish_streaming` for a zero-token
            /// result, not a real content delta. A test that only checks "a
            /// second frame arrived and was Ok" cannot tell these apart, so
            /// this asserts the frame's actual JSON payload shape: real code
            /// path yields `finish_reason: null` + a string `delta.content`;
            /// the mutated path yields a non-null `finish_reason` and no
            /// `content`, and the second `assert!` fails, so this test fails
            /// if the disconnect-stops-generation behavior regresses.
            #[tokio::test]
            async fn chat_completions_streaming_disconnect_stops_generation() {
                let req = ChatCompletionRequest {
                    model: Some("test-model".to_string()),
                    messages: vec![user_msg("hi")],
                    max_tokens: Some(NEAR_MAX_CONTEXT_TOKENS),
                    stream: Some(true),
                    ..bare_req()
                };

                let response = chat_completions_with_request(State(tiny_state()), req)
                    .await
                    .expect("streaming request must be accepted");
                assert_eq!(response.status(), StatusCode::OK);

                let mut body = response.into_body();

                // Frame 1: the role chunk, emitted unconditionally before any
                // generation output (`futures::stream::once(role_chunk).chain(body_stream)`).
                tokio::time::timeout(Duration::from_secs(10), body.frame())
                    .await
                    .expect("role chunk frame must arrive quickly")
                    .expect("role chunk frame must exist")
                    .expect("role chunk frame must be Ok");

                // Frame 2: must be a genuine content delta from the tiny
                // model's decode loop, not the finish chunk. A frame merely
                // *arriving* here does not prove generation ran: if
                // `cancel_guard` is no longer tied to the stream, it drops
                // (flipping `cancel_rx`) before the blocking task's
                // pre-prefill checkpoint, `generate_streaming_with_cancel`
                // returns `Interrupt` with zero tokens, and
                // `finish_streaming` sends `StreamMsg::Done` immediately --
                // which *also* produces a well-formed, `Ok` SSE frame here,
                // just one carrying `finish_reason: "length"` (the shared
                // mapping has no `"interrupt"` string; see
                // `lattice_inference::serve::finish_reason`'s doc comment)
                // and no `content` instead of a real delta. So this asserts
                // the frame's actual payload shape, not just its arrival.
                let frame = tokio::time::timeout(Duration::from_secs(10), body.frame())
                    .await
                    .expect(
                        "a second frame must arrive quickly -- if this times out, \
                         nothing at all was sent after the role chunk",
                    )
                    .expect("second frame must exist")
                    .expect("second frame must be Ok");
                let data = frame
                    .into_data()
                    .expect("second frame must carry data (not trailers)");
                let text = std::str::from_utf8(&data).expect("SSE frame must be UTF-8");
                let payload = text
                    .strip_prefix("data: ")
                    .unwrap_or(text)
                    .trim_end_matches(['\n', '\r']);
                let chunk: serde_json::Value =
                    serde_json::from_str(payload).expect("SSE frame must carry a JSON chunk");
                let choice = &chunk["choices"][0];
                assert!(
                    choice["finish_reason"].is_null(),
                    "second frame must be a content delta, not the finish chunk \
                     (generation was interrupted before producing any output, \
                     i.e. cancel_guard fired before the stream had a chance to \
                     run): {chunk}"
                );
                assert!(
                    choice["delta"]["content"].is_string(),
                    "second frame must carry delta.content (a real generated \
                     token), got: {chunk}"
                );

                // Client disconnect: drop the body. `cancel_guard` (captured
                // by `body_stream`'s `flat_map` closure) drops with it,
                // flipping the paired `cancel_rx` -- the CPU decode loop's
                // `generate_streaming_with_cancel` polls that flag at the top
                // of every iteration (already unit/mutation-tested in
                // `model/qwen35/generation.rs`) and stops within one step
                // instead of running toward `max_tokens` (900). This
                // test's own fast, deterministic completion (it does not hang
                // waiting on the now-unobservable background task) is the
                // practical proof: dropping never blocks, and the tiny
                // model's decode loop is fast enough on the blocking-thread
                // pool that a broken cancellation path would otherwise pin a
                // thread pool worker in an unbounded loop for the remainder
                // of the test process's life -- observable across this
                // binary's full test run as sporadic slowdowns/hangs in
                // later tests sharing the pool, not just this test in
                // isolation.
                drop(body);
            }
        }

        // -----------------------------------------------------------------------
        // Post-drop generator-side cancellation probe (ADR-080 C2):
        // `chat_completions_streaming_
        // disconnect_stops_generation` above proves guard retention BEFORE
        // the response is returned (frame 2 is a real content delta), but
        // does NOT prove `should_cancel` reaching the generator AFTER the
        // drop, independently of `on_token`'s own failed-send stop
        // condition -- the exact reverse mutation (`cancel_rx`
        // predicate replaced by `move || false`, `on_token`'s failed-send
        // path left intact) left that test green in 0.02s, because the
        // failed send alone stops a real decode loop just as fast as a
        // correctly wired cancellation would.
        //
        // This test isolates the two signals: `ModelBackend::CpuFakeGenerate`
        // (test-only, see its doc comment) sends exactly ONE delta -- so
        // `on_token` is never called again after that point -- then enters a
        // phase that polls ONLY `should_cancel`, in a loop that never touches
        // `on_token`/the reply channel at all, and reports which of two
        // outcomes ended that loop over a side channel `chat_completions`
        // itself never sees. Because the probe stops polling `on_token`
        // entirely after the first delta, `on_token`'s failed-send masking
        // effect (the exact thing that hid this bug from the original
        // test) cannot fire here -- only `should_cancel`'s own return value
        // can end the loop.
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        mod post_drop_cancellation_probe {
            use super::*;
            use http_body_util::BodyExt;
            use std::sync::Mutex;
            use std::time::Duration;

            /// One real delta, then a bounded should_cancel-only poll loop.
            /// `MAX_POLLS` * `POLL_INTERVAL` (10s) comfortably exceeds the
            /// test's own 10s await-completion timeout below, so a broken
            /// `should_cancel` wiring (the `move || false` mutation) makes
            /// the outer `tokio::time::timeout` fire first with a clear
            /// failure message, rather than this loop silently reporting
            /// "exhausted" first in a way that could be confused with a
            /// flake.
            const MAX_POLLS: usize = 4000;
            const POLL_INTERVAL: Duration = Duration::from_millis(5);

            /// Same rationale as `disconnect_cancellation`'s constant of the
            /// same value: the tiny test model's context window is a fixed
            /// 1024 tokens, so this stays comfortably under it while still
            /// being "effectively unbounded" relative to this test's own
            /// timeouts.
            const NEAR_MAX_CONTEXT_TOKENS: usize = 900;

            /// `should_cancel`'s reading taken after the test's `checkpoint1`
            /// signal (sent the instant `chat_completions(..).await`
            /// returns, before the test reads any SSE frame or drops
            /// anything) but before the `go` signal (sent only after
            /// `drop(body)`). Must be `false` in correctly-wired code:
            /// `cancel_guard` is still alive at this point (held by
            /// `body_stream`'s `flat_map` closure, itself alive because the
            /// test hasn't dropped the body/receiver yet), so `cancel_rx`
            /// cannot have flipped. A `true` reading here is the direct,
            /// timing-independent signature of the guard-capture-removed
            /// mutation: `cancel_guard` is then just an unused
            /// local that drops the instant `chat_completions` returns --
            /// gating this read on `checkpoint1` (rather than reading it
            /// immediately after `on_token`, which raced against
            /// `chat_completions`'s own return on an early version of this
            /// test and non-deterministically read `false` even under the
            /// mutation) guarantees the read happens causally after that
            /// return, so an unused `cancel_guard` has unconditionally
            /// already dropped by the time this reads it.
            #[derive(Debug, PartialEq)]
            struct ProbeOutcome {
                pre_drop_cancelled: bool,
                post_drop: PostDropOutcome,
            }

            #[derive(Debug, PartialEq)]
            enum PostDropOutcome {
                /// `should_cancel` returned `true` after this many polls
                /// following the test's `go` signal (sent only after
                /// `drop(body)`) -- the correctly-wired behavior.
                CancelledAfterPolls(usize),
                /// The poll budget was exhausted without `should_cancel`
                /// ever returning `true` -- the `move || false` mutation's
                /// signature (an unthreaded/replaced predicate that can
                /// never observe the drop).
                ExhaustedWithoutCancel,
            }

            /// Builds an `AppState` whose CPU streaming generation is the
            /// injected `generate` closure instead of the real tiny model's
            /// decode loop, via `ModelBackend::CpuFakeGenerate`. Tokenizer/
            /// context-window behavior still comes from a real tiny model
            /// (`tiny_zero_model()`), so request validation ahead of the
            /// streaming branch is unchanged from the other HTTP-level
            /// tests in this file.
            fn tiny_state_with_fake_cpu_generate(
                max_tokens_cap: usize,
                generate: impl Fn(
                    &str,
                    &lattice_inference::model::qwen35_config::GenerateConfig,
                    &mut dyn FnMut(&str) -> bool,
                    &mut dyn FnMut() -> bool,
                )
                    -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                + Send
                + Sync
                + 'static,
            ) -> AppState {
                let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
                AppState {
                    model: ModelBackend::CpuFakeGenerate {
                        model: Arc::new(model),
                        generate: Arc::new(generate),
                    },
                    default_max_tokens: max_tokens_cap,
                    max_tokens_cap,
                    model_id: "test-model".to_string(),
                    request_counter: Arc::new(AtomicU64::new(0)),
                }
            }

            #[tokio::test]
            async fn chat_completions_streaming_failure_emits_error_event() {
                let state = tiny_state_with_fake_cpu_generate(
                    64,
                    |_prompt, _cfg, on_token, _should_cancel| {
                        let _ = on_token("partial");
                        Err(lattice_inference::error::InferenceError::InvalidInput(
                            "blocked by grammar".to_string(),
                        ))
                    },
                );
                let req = ChatCompletionRequest {
                    model: Some("test-model".to_string()),
                    messages: vec![user_msg("hi")],
                    max_tokens: Some(64),
                    stream: Some(true),
                    ..bare_req()
                };

                let response = chat_completions_with_request(State(state), req)
                    .await
                    .expect("streaming request must be accepted");
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = response
                    .into_body()
                    .collect()
                    .await
                    .expect("SSE response body must be readable")
                    .to_bytes();
                let text = String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8");
                assert!(
                    text.contains("\"content\":\"partial\""),
                    "partial output must precede the error event; got: {text}"
                );
                let error_payload = text
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .find(|payload| payload.contains("\"error\""))
                    .expect("a failed generation must emit an SSE error payload");
                let error: serde_json::Value = serde_json::from_str(error_payload)
                    .expect("SSE error payload must be valid JSON");
                assert_eq!(error["error"]["type"], "server_error");
                assert_eq!(error["error"]["code"], "internal_error");
                assert!(
                    !text.contains("\"finish_reason\":\"stop\""),
                    "generation failure must not masquerade as a clean stop; got: {text}"
                );
            }

            /// Mutation-sensitive to BOTH known regressions
            /// independently, via a test<->generator handshake that removes
            /// the timing race a plain poll-and-time-it design would have
            /// (an early, timing-dependent version of this test observed
            /// `CancelledAfterPolls(1)` even under mutation (a), because
            /// `cancel_guard` -- unused once its capture is removed --
            /// drops at `chat_completions`'s own return, which can race
            /// ahead of or behind the generator's first poll depending on
            /// blocking-thread-pool scheduling):
            ///
            /// (a) removing `body_stream`'s
            ///     `let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;`
            ///     capture flips `cancel_rx` the instant `chat_completions`
            ///     returns -- before the test even reads its first SSE
            ///     frame, let alone drops the body. `pre_drop_cancelled`
            ///     (read BEFORE the generator waits on the `go` signal,
            ///     which the test only sends after `drop(body)`) captures
            ///     exactly this: `true` here is impossible under correct
            ///     wiring regardless of scheduling, since `cancel_guard` is
            ///     provably still alive at that point.
            /// (b) replacing the CPU `should_cancel` predicate
            ///     (`move || *cancel_rx.borrow()`) with `move || false`
            ///     means the post-`go` poll loop never observes `true`, so
            ///     it exhausts `MAX_POLLS` and reports
            ///     `PostDropOutcome::ExhaustedWithoutCancel`.
            #[tokio::test]
            async fn chat_completions_streaming_disconnect_cancellation_reaches_generator_post_drop()
             {
                let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel::<ProbeOutcome>();
                let outcome_tx = Mutex::new(Some(outcome_tx));
                let (checkpoint1_tx, checkpoint1_rx) = std::sync::mpsc::channel::<()>();
                let checkpoint1_rx = Mutex::new(Some(checkpoint1_rx));
                let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
                let go_rx = Mutex::new(Some(go_rx));

                let state = tiny_state_with_fake_cpu_generate(
                    NEAR_MAX_CONTEXT_TOKENS,
                    move |_prompt, _cfg, on_token, should_cancel| {
                        // One real delta -- exactly what the disconnect test
                        // above reads as its second frame -- so
                        // `chat_completions`'s SSE framing has genuine
                        // content before this probe phase begins. Queued
                        // into the reply channel immediately; the test can
                        // read it as frame 2 regardless of whether this
                        // thread is later blocked waiting on `checkpoint1`.
                        on_token("probe");

                        // Block until the test confirms `chat_completions`
                        // itself has already returned (see `pre_drop_cancelled`'s
                        // doc comment on why this ordering, not an
                        // immediate post-`on_token` read, is what makes the
                        // next line's reading timing-independent).
                        if let Some(rx) = checkpoint1_rx.lock().unwrap().take() {
                            let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
                        }
                        let pre_drop_cancelled = should_cancel();

                        // Block until the test signals it has dropped the
                        // body (or give up after 10s so a broken handshake
                        // fails this test instead of hanging the process).
                        if let Some(rx) = go_rx.lock().unwrap().take() {
                            let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
                        }

                        let mut polls = 0usize;
                        let post_drop = loop {
                            if should_cancel() {
                                break PostDropOutcome::CancelledAfterPolls(polls);
                            }
                            if polls >= MAX_POLLS {
                                break PostDropOutcome::ExhaustedWithoutCancel;
                            }
                            polls += 1;
                            std::thread::sleep(POLL_INTERVAL);
                        };
                        if let Some(tx) = outcome_tx.lock().unwrap().take() {
                            let _ = tx.send(ProbeOutcome {
                                pre_drop_cancelled,
                                post_drop,
                            });
                        }
                        Ok(GenerateOutput {
                            text: "probe".to_string(),
                            token_ids: vec![],
                            prompt_tokens: 1,
                            generated_tokens: 1,
                            stopped: false,
                            stop_reason: Some(
                                lattice_inference::stop_reason::StopReason::Interrupt,
                            ),
                            token_logprobs: vec![],
                        })
                    },
                );

                let req = ChatCompletionRequest {
                    model: Some("test-model".to_string()),
                    messages: vec![user_msg("hi")],
                    max_tokens: Some(NEAR_MAX_CONTEXT_TOKENS),
                    stream: Some(true),
                    ..bare_req()
                };
                let response = chat_completions_with_request(State(state), req)
                    .await
                    .expect("streaming request must be accepted");
                // `chat_completions` has now unconditionally returned, so an
                // unused `cancel_guard` (mutation (a)) has already dropped;
                // signal the generator it may take its `pre_drop_cancelled`
                // reading.
                let _ = checkpoint1_tx.send(());
                let mut body = response.into_body();

                // Frame 1: role chunk.
                tokio::time::timeout(Duration::from_secs(10), body.frame())
                    .await
                    .expect("role chunk frame must arrive quickly")
                    .expect("role chunk frame must exist")
                    .expect("role chunk frame must be Ok");

                // Frame 2: the fake generator's one real delta.
                tokio::time::timeout(Duration::from_secs(10), body.frame())
                    .await
                    .expect("delta frame must arrive quickly")
                    .expect("delta frame must exist")
                    .expect("delta frame must be Ok");

                // Client disconnect: drop the body. The fake generator is
                // waiting on `go_rx`, having already called `on_token` and
                // taken its `pre_drop_cancelled` reading.
                drop(body);
                let _ = go_tx.send(());

                let outcome = tokio::time::timeout(Duration::from_secs(10), outcome_rx)
                    .await
                    .expect(
                        "the fake generator's post-drop probe phase must signal \
                         completion within 10s of the client disconnecting -- a \
                         timeout here means should_cancel never observed the \
                         disconnect at all",
                    )
                    .expect("probe outcome sender must not be dropped without sending");

                assert!(
                    !outcome.pre_drop_cancelled,
                    "should_cancel must read false before the test has dropped the \
                     response body -- a true reading here means cancel_guard had \
                     already dropped (e.g. because it is no longer tied to the \
                     stream's lifetime) well before any disconnect happened: \
                     {outcome:?}"
                );
                assert!(
                    matches!(outcome.post_drop, PostDropOutcome::CancelledAfterPolls(_)),
                    "should_cancel must return true within the poll budget after \
                     the test drops the response body -- exhausting the budget \
                     means the disconnect signal never reached the generator at \
                     all: {outcome:?}"
                );
            }
        }

        // -----------------------------------------------------------------------
        // Streaming context-overflow status parity (ADR-080 C2): `lattice.rs` already gets this right
        // structurally -- `prepare_chat_request`'s context-window preflight
        // (`check_context_window`) runs unconditionally, before
        // `chat_completions` ever branches on `req.stream` -- so a `stream:
        // true` request that overflows the model's context window returns
        // HTTP 400 `context_length_exceeded` before any SSE stream is ever
        // built. `cm_serve_context_window_checked_before_stop_parsing` above
        // already pins the underlying cascade ordering as a pure function;
        // this drives the SAME contract through the real `Router`.
        //
        // ADR-080 C2: this now
        // builds its request from `lattice_inference::serve`'s shared
        // `OVERFLOW_PARITY_*` constants -- the SAME body/limits
        // `lattice_serve.rs`'s `real_router_overflow_parity` module drives
        // through its own real router and real worker -- so the two really
        // are the identical request the old doc comment here merely
        // claimed. The tiny test model's context window
        // (`test_support::tiny_zero_model`'s `max_position_embeddings`) is
        // fixed at `OVERFLOW_PARITY_CONTEXT_WINDOW` (1024) precisely so this
        // side's "effective context limit" matches the daemon side's
        // explicitly-configured `AppState.model_max_context` of the same
        // value.
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        mod streaming_context_overflow {
            use super::*;
            use lattice_inference::serve::{
                OVERFLOW_PARITY_MAX_TOKENS_CAP, OVERFLOW_PARITY_REQUEST_BODY,
            };
            use tower::ServiceExt as _;

            #[tokio::test]
            async fn chat_completions_streaming_context_overflow_returns_400_before_committing_sse()
            {
                let body = axum::body::Body::from(OVERFLOW_PARITY_REQUEST_BODY.to_string());
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(body)
                    .expect("fixture request must build");
                let response = router(tiny_state(OVERFLOW_PARITY_MAX_TOKENS_CAP))
                    .oneshot(request)
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "an over-context stream:true request must be rejected with HTTP \
                     400 before any SSE stream is committed, matching \
                     lattice_serve.rs's equivalent preflight for the identical \
                     shared-fixture request body"
                );
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("error response body must be readable");
                let value: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("error response must be JSON");
                assert_eq!(value["error"]["code"], "context_length_exceeded");
            }
        }

        // -----------------------------------------------------------------------
        // Message-flood bound: proves the fix at the real HTTP layer, not just
        // at `serve::contract`'s own unit-test level -- the router's
        // `chat_completions` entry point must reject a body with more than
        // `MAX_MESSAGE_COUNT` tiny messages.
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        mod message_flood {
            use super::*;
            use lattice_inference::serve::contract::MAX_MESSAGE_COUNT;
            use tower::ServiceExt as _;

            #[tokio::test]
            async fn chat_completions_rejects_message_flood() {
                // One more message than the bound, each as small as the wire
                // format allows: comfortably under the 1 MiB body cap, but
                // tens of thousands of entries. The message-count bound is
                // enforced inline while `ChatCompletionRequest::messages`
                // deserializes, so this never allocates a `Vec<Message>`
                // entry per message before rejecting -- see this test's
                // sibling unit tests in `serve::contract` for direct coverage
                // of that deserializer.
                let messages: Vec<String> = (0..MAX_MESSAGE_COUNT + 1)
                    .map(|_| r#"{"role":"user","content":""}"#.to_string())
                    .collect();
                let body = format!(
                    r#"{{"model":"test-model","messages":[{}]}}"#,
                    messages.join(",")
                );
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .expect("fixture request must build");
                let response = router(tiny_state(64))
                    .oneshot(request)
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("error response body must be readable");
                let value: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("error response must be JSON");
                assert_eq!(value["error"]["code"], "invalid_request_body");
            }
        }

        // -----------------------------------------------------------------------
        // Content-Type precedence: an invalid-MIME request must be rejected by
        // `require_json_content_type` before the body is ever read as JSON.
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        mod content_type_precedence {
            use super::*;
            use tower::ServiceExt as _;

            /// A body that is not valid JSON at all. Combined with a
            /// non-JSON `Content-Type`, this distinguishes ordering: if the
            /// Content-Type guard runs first, the response is 415
            /// `unsupported_media_type`; if it were reordered to run after
            /// the body is parsed, this body would instead fail JSON
            /// parsing first and surface as 400 `invalid_request_body`.
            #[tokio::test]
            async fn invalid_content_type_rejected_before_body_is_parsed() {
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "text/plain")
                    .body(axum::body::Body::from("this is not json"))
                    .expect("fixture request must build");
                let response = router(tiny_state(64))
                    .oneshot(request)
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("error response body must be readable");
                let value: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("error response must be JSON");
                assert_eq!(value["error"]["code"], "unsupported_media_type");
            }
        }

        // -----------------------------------------------------------------------
        // Cross-binary `/v1/chat/completions` parity table (ADR-080 C2):
        // drives every fixture body in
        // `lattice_inference::serve::CHAT_COMPLETIONS_PARITY_CASES` through
        // THIS binary's real `Router` via `tower::ServiceExt::oneshot`, and
        // compares the resulting status + error code against the case's
        // `lattice`-side expectation. `lattice_serve.rs`'s own test module
        // runs the SAME table against its own router, asserting the
        // `lattice_serve`-side expectation -- together the two prove
        // same-input parity (or a documented, intentional divergence) at the
        // real HTTP layer, not just in each binary's private validation
        // helpers. Gated behind `test-utils` for the same reason as
        // `disconnect_cancellation`: `router()` needs a real `AppState`,
        // which needs a real (tiny) CPU model.
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        mod parity_table {
            use super::*;
            use lattice_inference::StopReason;
            use lattice_inference::serve::{
                BASELINE_CANNED_COMPLETION_TOKENS, BASELINE_CANNED_PROMPT_TOKENS,
                BASELINE_CANNED_TEXT, Binary, CHAT_COMPLETIONS_PARITY_CASES, ExpectedResponse,
                check_sse_events,
            };
            use tower::ServiceExt as _;

            /// Small enough that `max_tokens_over_cap_reject_vs_clamp`'s
            /// 999999 genuinely exceeds both `default_max_tokens` and
            /// `max_tokens_cap`.
            const CAP: usize = 64;

            /// Deterministic CPU generation seam for every `Json`/`Sse`
            /// row (issue #828): the real request-parse/normalize/
            /// `GenerateConfig`-build/handler/serialization path all still
            /// runs unmodified -- only the actual model forward pass is
            /// replaced, via the SAME `ModelBackend::CpuFakeGenerate`
            /// injection seam the disconnect-cancellation probe uses.
            /// Content deltas are pushed through `on_token` (what the
            /// streaming arm reads) AND the returned `GenerateOutput.text`
            /// carries the identical concatenated text (what the
            /// non-streaming arm reads) so one closure serves both shapes.
            fn baseline_fake_state(max_tokens_cap: usize) -> AppState {
                let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
                #[allow(clippy::type_complexity)]
                let generate: Arc<
                    dyn Fn(
                            &str,
                            &lattice_inference::model::qwen35_config::GenerateConfig,
                            &mut dyn FnMut(&str) -> bool,
                            &mut dyn FnMut() -> bool,
                        )
                            -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                        + Send
                        + Sync,
                > = Arc::new(|_prompt, _cfg, on_token, _should_cancel| {
                    for chunk in ["hello", " world"] {
                        if !on_token(chunk) {
                            break;
                        }
                    }
                    Ok(GenerateOutput {
                        text: BASELINE_CANNED_TEXT.to_string(),
                        token_ids: vec![1, 2],
                        prompt_tokens: BASELINE_CANNED_PROMPT_TOKENS as usize,
                        generated_tokens: BASELINE_CANNED_COMPLETION_TOKENS as usize,
                        stopped: true,
                        stop_reason: Some(StopReason::Eos),
                        token_logprobs: vec![],
                    })
                });
                AppState {
                    model: ModelBackend::CpuFakeGenerate {
                        model: Arc::new(model),
                        generate,
                    },
                    default_max_tokens: max_tokens_cap,
                    max_tokens_cap,
                    model_id: "test-model".to_string(),
                    request_counter: Arc::new(AtomicU64::new(0)),
                }
            }

            #[tokio::test]
            async fn chat_completions_matches_shared_parity_table() {
                for case in CHAT_COMPLETIONS_PARITY_CASES {
                    let expected = case.expected(Binary::Lattice);
                    // Error-shaped rows never reach generation (rejected at
                    // validation), so they keep using the plain real tiny
                    // model exactly as before #828; only the new `Json`/
                    // `Sse` rows need the deterministic generation seam.
                    let app = match expected {
                        ExpectedResponse::Error { .. } => router(tiny_state(CAP)),
                        ExpectedResponse::Json { .. } | ExpectedResponse::Sse { .. } => {
                            router(baseline_fake_state(CAP))
                        }
                    };
                    let request = axum::http::Request::builder()
                        .method(case.method)
                        .uri(case.path)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(case.body.build()))
                        .expect("fixture request must build");
                    let response = app
                        .oneshot(request)
                        .await
                        .expect("router must produce a response, not a transport error");

                    let status = response.status().as_u16();
                    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .expect("response body reads");
                    let text = String::from_utf8_lossy(&body);

                    assert_eq!(
                        status,
                        expected.status(),
                        "case '{}': expected status {}, got {status} (body: {text})",
                        case.name,
                        expected.status(),
                    );

                    match expected {
                        ExpectedResponse::Error { code, .. } => {
                            let value: serde_json::Value = serde_json::from_slice(&body)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "case '{}': non-2xx response body must be the shared \
                                         error envelope JSON: {e} (body: {text})",
                                        case.name,
                                    )
                                });
                            assert_eq!(
                                value["error"]["code"], code,
                                "case '{}': expected error code '{code}', got {} \
                                 (full body: {value})",
                                case.name, value["error"]["code"]
                            );
                        }
                        ExpectedResponse::Json { fields, .. } => {
                            let value: serde_json::Value = serde_json::from_slice(&body)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "case '{}': 2xx response body must be JSON: {e} \
                                         (body: {text})",
                                        case.name,
                                    )
                                });
                            for field in fields {
                                field.check(&value).unwrap_or_else(|e| {
                                    panic!("case '{}': field check failed: {e}", case.name)
                                });
                            }
                        }
                        ExpectedResponse::Sse { events, .. } => {
                            check_sse_events(&text, events).unwrap_or_else(|e| {
                                panic!("case '{}': SSE check failed: {e}", case.name)
                            });
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Production-adapter observation (issue #828): proves the shared
        // `ProductionAdapterObservation`/`GenerateConfigSnapshot` types
        // actually capture what THIS binary's real `chat_completions` ->
        // `prepare_chat_request` -> `GenerateConfig` construction produces,
        // not a value the test independently reconstructs. The injected
        // `CpuFakeGenerate` closure below runs strictly BELOW that real
        // path -- it records the `&GenerateConfig`/`&str` prompt it was
        // actually called with, then returns a canned result; it never
        // recomputes `build_cfg`/`validate_temperature`/etc. itself.
        //
        // DISPUTED (issue #828):
        // this observation captures `rendered_prompt`, not `messages`, and
        // that is the real shape of this seam, not an omission. `chat_completions`
        // computes `to_chat_messages(&req.messages)` (the normalized message
        // list) unconditionally whenever `feature = "metal-gpu"` is compiled
        // in, but that value is consumed ONLY by the `ModelBackend::Metal`
        // match arm (`handle.generate_streaming[_with_cancel](chat_messages,
        // ...)`); the `ModelBackend::Cpu`/`CpuFakeGenerate` arms this test
        // seam exercises never receive it -- their real `generate`/
        // `generate_streaming_with_cancel` calls take only `(&prompt,
        // &gen_cfg, ...)`. This mirrors `ProductionAdapterObservation`'s own
        // documented contract in `serve/mod.rs` ("exactly one of
        // `rendered_prompt`/`messages` is `Some` per capture, reflecting
        // which shape that binary's real adapter actually receives, not a
        // missing capture").
        //
        // Observing `messages` at the CPU seam authentically (not by
        // re-deriving `to_chat_messages` independently in the test, which
        // would be tautological -- exactly the bug this
        // module was written to fix) would require a `MetalFakeGenerate`
        // test double for `ModelBackend::Metal`. `MetalHandle::spawn`
        // hard-requires loading a real Q4 model directory onto a real Metal
        // GPU worker thread (`MetalQwen35State::from_q4_dir`) -- there is no
        // model-agnostic seam there the way `CpuFakeGenerate` mirrors
        // `ModelBackend::Cpu`. Building one would mean adding a new
        // production `ModelBackend` variant and mocking the async engine
        // handle's job-channel protocol: real production-code surface
        // expansion, not a test-only capture. That is out of scope for this
        // fix round; tracked as a follow-up if a Metal-path observation is
        // wanted (would need its own issue -- #828's fixture data and CI
        // environment target the CPU/tiny-tokenizer seam only).
        // -----------------------------------------------------------------------
        #[cfg(feature = "test-utils")]
        mod production_adapter_observation {
            use super::*;
            use lattice_inference::serve::{
                ExpectedObservation, GenerateConfigSnapshot,
                OBSERVATION_GOLDEN_USER_HI_THERE_CHATML, ProductionAdapterObservation,
                assert_observation_matches,
            };
            use std::sync::Mutex;
            use tower::ServiceExt as _;

            /// Builds the fixture state + fires the fixed `{"messages":[{"role":
            /// "user","content":"hi there"}],"temperature":1.3,"top_p":0.55,
            /// "seed":7,"max_tokens":9}` request against a real router, with the
            /// injected `CpuFakeGenerate` closure recording a
            /// `ProductionAdapterObservation` -- strictly below the real
            /// request-parse/`format_chat_template`/`GenerateConfig`-construction path
            /// (issue #828). `stopped` is threaded through a single local
            /// variable into both the recorded observation and the returned
            /// `GenerateOutput`, so a caller of this helper can vary it and prove
            /// the observation genuinely mirrors what the seam returned rather
            /// than an independent hardcoded literal.
            async fn run_observed(stopped: bool) -> ProductionAdapterObservation {
                let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
                let tokenizer = model.tokenizer().clone();
                let observed: Arc<Mutex<Option<ProductionAdapterObservation>>> =
                    Arc::new(Mutex::new(None));
                let observed_for_closure = Arc::clone(&observed);
                #[allow(clippy::type_complexity)]
                let generate: Arc<
                    dyn Fn(
                            &str,
                            &lattice_inference::model::qwen35_config::GenerateConfig,
                            &mut dyn FnMut(&str) -> bool,
                            &mut dyn FnMut() -> bool,
                        )
                            -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                        + Send
                        + Sync,
                > = Arc::new(move |prompt, cfg, _on_token, _should_cancel| {
                    let prompt_tokens = tokenizer.tokenize(prompt).real_length;
                    *observed_for_closure
                        .lock()
                        .expect("observation mutex poisoned") =
                        Some(ProductionAdapterObservation {
                            rendered_prompt: Some(prompt.to_string()),
                            messages: None,
                            gen_cfg: GenerateConfigSnapshot::from(cfg),
                            prompt_tokens,
                            stopped,
                        });
                    Ok(GenerateOutput {
                        text: "ok".to_string(),
                        token_ids: vec![1],
                        prompt_tokens,
                        generated_tokens: 1,
                        stopped,
                        stop_reason: if stopped {
                            Some(lattice_inference::StopReason::Eos)
                        } else {
                            None
                        },
                        token_logprobs: vec![],
                    })
                });
                let state = AppState {
                    model: ModelBackend::CpuFakeGenerate {
                        model: Arc::new(model),
                        generate,
                    },
                    default_max_tokens: 64,
                    max_tokens_cap: 64,
                    model_id: "test-model".to_string(),
                    request_counter: Arc::new(AtomicU64::new(0)),
                };
                let body = r#"{"model":"test-model","messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}"#;
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .expect("fixture request must build");
                let response = router(state)
                    .oneshot(request)
                    .await
                    .expect("router must produce a response, not a transport error");
                assert_eq!(response.status(), StatusCode::OK);

                observed
                    .lock()
                    .expect("observation mutex poisoned")
                    .clone()
                    .expect("the injected generate closure must have recorded an observation")
            }

            /// The `GenerateConfig` `lattice.rs`'s real `chat_completions` ->
            /// `prepare_chat_request`/`build_cfg`-equivalent construction (see
            /// its `let gen_cfg = ...` literal in this file) must produce for the
            /// fixed request body `run_observed` sends: every explicitly-set
            /// field mirrors the request, every other field is
            /// `GenerateConfig::default()` -- exactly like production's own
            /// `..Default::default()` tail.
            fn expected_gen_cfg() -> GenerateConfigSnapshot {
                GenerateConfigSnapshot::from(
                    &lattice_inference::model::qwen35_config::GenerateConfig {
                        max_new_tokens: 9,
                        temperature: 1.3,
                        top_p: 0.55,
                        seed: Some(7),
                        stop_strings: vec![],
                        logprobs: None,
                        ..Default::default()
                    },
                )
            }

            #[tokio::test]
            async fn chat_completions_non_streaming_observation_captures_real_config_and_prompt() {
                let obs = run_observed(true).await;
                let expected_prompt_tokens = {
                    let tokenizer =
                        lattice_inference::model::qwen35::test_support::tiny_zero_model()
                            .tokenizer()
                            .clone();
                    tokenizer
                        .tokenize(OBSERVATION_GOLDEN_USER_HI_THERE_CHATML)
                        .real_length
                };
                assert_observation_matches(
                    &obs,
                    &ExpectedObservation {
                        gen_cfg: expected_gen_cfg(),
                        rendered_prompt: Some(OBSERVATION_GOLDEN_USER_HI_THERE_CHATML),
                        messages: None,
                        prompt_tokens: expected_prompt_tokens,
                        stopped: true,
                    },
                );
            }

            /// Proves `ProductionAdapterObservation::stopped` is genuinely
            /// derived from what the generation seam returned, not an
            /// independent hardcoded literal (the
            /// pre-fix `lattice_serve.rs` observation stored `stopped: true`
            /// unconditionally). Running the exact same request through
            /// `run_observed(false)` must observe `stopped == false`.
            #[tokio::test]
            async fn chat_completions_non_streaming_observation_captures_real_stopped_false() {
                let obs = run_observed(false).await;
                assert!(
                    !obs.stopped,
                    "observation must report the seam's actual stopped=false, not a hardcoded true"
                );
            }
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
            max_pending,
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
                            max_pending,
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
                        let _ = max_pending;
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
        Command::PruneScore { args } => match prune_score::run(&args) {
            Ok(true) => {}
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    }
}
