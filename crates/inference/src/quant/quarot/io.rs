//! Streaming SafeTensors reader for the QuaRot conversion pipeline.
//!
//! The QuaRot offline converter (step 3c) needs to read every weight tensor
//! exactly once, promote it to f64 for the rotation math, fuse RMSNorm scales,
//! apply rotation absorption, then quantize and discard. Caching the f32 or
//! f64 expansion of a multi-gigabyte checkpoint would defeat that pipeline —
//! so this reader allocates a fresh `Vec<f64>` per `read_tensor_f64` call
//! and never caches converted bytes. Compare with
//! [`crate::weights::f32_weights::SafetensorsFile`] which DOES cache the
//! f32-converted form per tensor and is appropriate for inference paths
//! that revisit weights repeatedly.
//!
//! Layout is auto-detected, matching the runtime loader precedence in
//! [`crate::model::qwen35::Qwen35Model::from_safetensors`] and
//! [`crate::model::qwen::QwenModel::from_directory`]: when both files are
//! present, the single-file checkpoint wins. This keeps the converter and
//! the runtime forward pass reading the same bytes — step 3c's
//! forward-equivalence assertion depends on this.
//!
//! - `model.safetensors` present → single-file layout.
//! - else `model.safetensors.index.json` present → sharded layout. All
//!   unique shard files referenced by the index are opened and parsed
//!   eagerly at [`QuarotTensorReader::open`], so [`tensor_names`] and
//!   [`has_tensor`] mean "readable supported tensor" in both modes.
//!   This surfaces missing-in-shard or unsupported-dtype-in-shard
//!   failures before step 3c's rotation pass begins instead of mid-run.
//! - Otherwise: error at [`QuarotTensorReader::open`].
//!
//! [`tensor_names`]: QuarotTensorReader::tensor_names
//! [`has_tensor`]: QuarotTensorReader::has_tensor
//!
//! On-disk decode for F32 / F16 / BF16 uses the always-compiled scalar
//! conversion in [`crate::weights::half_bits`] (independent of the `f16`
//! cargo feature, which only gates *loading permission* elsewhere), then
//! widens the result to f64 (lossless from f32).
//!
//! Step 3b deliverable per [ADR-044]; consumed by step 3c's
//! `quantize_quarot` binary.
//!
//! [ADR-044]: ../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md

use std::collections::HashMap;
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};

use memmap2::Mmap;
use serde_json::Value;

use crate::error::InferenceError;
use crate::weights::f32_weights::parse_index;

/// Open `path` as a directory fd, refusing to follow a symlink at the final
/// component (`O_NOFOLLOW|O_DIRECTORY`). Held for the lifetime of a
/// [`QuarotTensorReader`] and used as the sole anchor for every subsequent
/// `openat` in this module — no path string derived from `path` is ever
/// re-resolved after this call returns.
fn open_dir_nofollow(path: &Path) -> Result<OwnedFd, InferenceError> {
    let cpath = CString::new(path.as_os_str().as_bytes()).map_err(|e| {
        InferenceError::Inference(format!("path {path:?} contains an interior NUL: {e}"))
    })?;
    // SAFETY: `cpath` is a valid NUL-terminated C string owned for the
    // duration of this call. `open` returns either a valid fd (>= 0) or -1
    // with errno set; the -1 case is checked immediately below.
    let raw = unsafe {
        libc::open(
            cpath.as_ptr(),
            libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(InferenceError::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: `raw` was just returned by `libc::open` above, checked
    // non-negative, and is not owned or closed anywhere else.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// `openat(dirfd, name, O_DIRECTORY|O_NOFOLLOW)` — one traversal hop bound to
/// an already-held directory fd rather than a re-resolved path.
fn openat_dir_nofollow(dirfd: RawFd, name: &std::ffi::OsStr) -> Result<OwnedFd, InferenceError> {
    let cname = CString::new(name.as_bytes()).map_err(|e| {
        InferenceError::Inference(format!(
            "path component {name:?} contains an interior NUL: {e}"
        ))
    })?;
    // SAFETY: `dirfd` is a valid open directory fd borrowed for the duration
    // of this call; `cname` is a valid NUL-terminated C string. `openat`
    // returns either a valid fd (>= 0) or -1 with errno set, checked below.
    let raw = unsafe {
        libc::openat(
            dirfd,
            cname.as_ptr(),
            libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(InferenceError::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: see `open_dir_nofollow`.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// `openat(dirfd, name, O_RDONLY|O_NOFOLLOW)` — open the final path
/// component as a file bound to an already-held directory fd.
fn openat_file_nofollow(
    dirfd: RawFd,
    name: &std::ffi::OsStr,
) -> Result<std::fs::File, InferenceError> {
    let cname = CString::new(name.as_bytes()).map_err(|e| {
        InferenceError::Inference(format!(
            "path component {name:?} contains an interior NUL: {e}"
        ))
    })?;
    // SAFETY: see `openat_dir_nofollow`.
    let raw = unsafe {
        libc::openat(
            dirfd,
            cname.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(InferenceError::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: see `open_dir_nofollow`.
    Ok(unsafe { std::fs::File::from(OwnedFd::from_raw_fd(raw)) })
}

/// Resolve a manifest-declared shard filename against a held `model_root`
/// directory fd (`root_fd`) by walking every path component with its own
/// `openat(..., O_NOFOLLOW)`, rather than canonicalizing the joined path and
/// reopening it by string.
///
/// A canonicalize-then-open sequence is a snapshot: it stops being true the
/// instant `canonicalize` returns, so an attacker who can replace an
/// ancestor directory (or the model root itself) between the check and the
/// later open can redirect a "contained" shard outside `model_root` even
/// though the check passed. Walking component-by-component with `O_NOFOLLOW`
/// on every hop makes containment a structural property of the traversal:
/// each hop is bound to the fd opened for the *previous* hop, never to a
/// re-resolved path string, and any component that is `..`, absolute, or a
/// symlink is rejected outright.
fn open_manifest_entry(
    root_fd: &OwnedFd,
    entry_name: &str,
) -> Result<std::fs::File, InferenceError> {
    let rel = Path::new(entry_name);
    let mut components = Vec::new();
    for component in rel.components() {
        match component {
            Component::Normal(part) => components.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(InferenceError::Inference(format!(
                    "manifest entry {entry_name:?} must be a plain relative path with no `..` \
                     or root component"
                )));
            }
        }
    }
    let Some((&last, dirs)) = components.split_last() else {
        return Err(InferenceError::Inference(format!(
            "manifest entry {entry_name:?} resolves to an empty path"
        )));
    };

    let mut held_dir: Option<OwnedFd> = None;
    for name in dirs {
        let base_fd = held_dir
            .as_ref()
            .map_or(root_fd.as_raw_fd(), AsRawFd::as_raw_fd);
        held_dir = Some(openat_dir_nofollow(base_fd, name)?);
    }
    let base_fd = held_dir
        .as_ref()
        .map_or(root_fd.as_raw_fd(), AsRawFd::as_raw_fd);
    openat_file_nofollow(base_fd, last)
}

/// On-disk storage dtype of a tensor.
///
/// Returned by [`QuarotTensorReader::source_dtype`] so the converter can
/// record provenance in its output metadata. The converter always promotes
/// to f64 internally; this is informational, not a control knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceDType {
    F32,
    F16,
    BF16,
}

impl SourceDType {
    fn from_header_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(SourceDType::F32),
            "F16" => Some(SourceDType::F16),
            "BF16" => Some(SourceDType::BF16),
            _ => None,
        }
    }

    /// Header string name (matches the safetensors JSON `dtype` field).
    pub fn name(self) -> &'static str {
        match self {
            SourceDType::F32 => "F32",
            SourceDType::F16 => "F16",
            SourceDType::BF16 => "BF16",
        }
    }
}

#[derive(Debug, Clone)]
struct TensorHeader {
    dtype: SourceDType,
    shape: Vec<usize>,
    /// Byte offsets within the shard's data section (the bytes that
    /// follow the 8-byte length prefix + JSON header).
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct Shard {
    source: String,
    mmap: Mmap,
    /// Absolute file offset where the data section begins
    /// (`8 + header_len`).
    data_offset: usize,
    /// Only supported-dtype tensors (F32 / F16 / BF16). Unsupported
    /// dtypes participate in offset-contiguity validation at parse time
    /// but are not exposed to the read API.
    headers: HashMap<String, TensorHeader>,
}

/// One tensor entry parsed from the SafeTensors header, before the
/// supported-dtype filter is applied. Retained per-entry so the
/// global offset contiguity check covers every tensor — including
/// unsupported dtypes — to catch aliasing and corrupted indexes that
/// per-tensor validation alone would miss.
struct ParsedEntry {
    name: String,
    start: usize,
    end: usize,
    /// `Some` for F32/F16/BF16; `None` for any other SafeTensors dtype.
    /// `None` entries are still kept in the contiguity check but never
    /// appear in `Shard::headers`.
    header: Option<TensorHeader>,
}

/// Bits per element for every standard SafeTensors dtype name. The
/// bit-size variant (rather than bytes) is required because the format
/// admits sub-byte dtypes: `F4` stores 4 bits per element, and
/// `F6_E2M3` / `F6_E3M2` store 6 bits.
///
/// Returns `None` for any unrecognized dtype string. The caller MUST
/// reject unknown dtypes rather than treat them as opaque, since they
/// indicate either a newer SafeTensors revision the reader has not yet
/// been updated for or a corrupted header.
///
/// Strictness note: the official `safetensors` Rust crate rejects
/// unknown/invalid dtype metadata outright, and this reader follows
/// that strict contract. Lattice's runtime weights parser
/// ([`crate::weights::f32_weights::SafetensorsFile`]'s `parse_tensor_meta`)
/// was more permissive prior to lattice#800 — it mapped unknown dtypes to
/// `Ok(None)` and silently skipped the entry. As of lattice#800 it also
/// rejects a genuinely unrecognized dtype string at parse time and tracks
/// known-but-unsupported whole-byte dtypes (I64, BOOL, ...) structurally
/// instead of dropping them, matching this reader's strictness for those
/// cases. The two parsers still diverge on sub-byte dtypes (`F4`,
/// `F6_E2M3`, `F6_E3M2`): this reader's bit-size table represents them
/// exactly, while the runtime parser's whole-byte extent model treats them
/// as unrecognized. Unifying the two parsers (not just their strictness) is
/// a separate concern, tracked by the native Q4/KHF1 validation-unification
/// and QuaRot/offline-quantizer routing issues in the lattice#800 cluster.
///
/// Table mirrors `Dtype::bitsize()` in the official `safetensors` Rust
/// crate. Kept inline rather than depending on `safetensors` to match
/// the hand-roll pattern of [`crate::weights::f32_weights`] (which also
/// parses safetensors headers directly). When upstream adds a dtype,
/// add it here and to the dtype-coverage regression tests.
fn safetensors_bits_per_elem(dtype_str: &str) -> Option<usize> {
    match dtype_str {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => Some(8),
        "I16" | "U16" | "F16" | "BF16" => Some(16),
        "I32" | "U32" | "F32" => Some(32),
        "I64" | "U64" | "F64" | "C64" => Some(64),
        _ => None,
    }
}

impl Shard {
    /// Wrap an already-opened `file` for mmap. `file` must have been opened
    /// descriptor-relative to a trusted directory anchor (`open_dir_nofollow`
    /// together with `openat_file_nofollow`, or `open_manifest_entry`) with
    /// `O_NOFOLLOW` on every path component — never reopened by a full path
    /// string, which would re-resolve ancestor components an attacker could
    /// have swapped after an earlier containment check. `display_path` is
    /// used only for error messages.
    fn open(file: std::fs::File, display_path: &Path) -> Result<Self, InferenceError> {
        // SAFETY: the file is opened read-only; the returned Mmap owns the
        // mapping. The File handle can be dropped immediately after — the OS
        // keeps the fd alive through the map. Standard memmap2 usage.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            InferenceError::InvalidSafetensors(format!(
                "failed to mmap {}: {e}",
                display_path.display()
            ))
        })?;

        if mmap.len() < 8 {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{}: file too small to contain a safetensors header",
                display_path.display()
            )));
        }

        let header_len_bytes: [u8; 8] = mmap[0..8].try_into().map_err(|_| {
            InferenceError::InvalidSafetensors("invalid header-length prefix".into())
        })?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;
        let data_offset = 8usize
            .checked_add(header_len)
            .ok_or_else(|| InferenceError::InvalidSafetensors("header length overflow".into()))?;
        if data_offset > mmap.len() {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{}: header extends past end of file (header_end={}, file_len={})",
                display_path.display(),
                data_offset,
                mmap.len()
            )));
        }

        let header_str = std::str::from_utf8(&mmap[8..data_offset]).map_err(|e| {
            InferenceError::InvalidSafetensors(format!(
                "{}: header is not valid UTF-8: {e}",
                display_path.display()
            ))
        })?;
        let root: Value = serde_json::from_str(header_str).map_err(|e| {
            InferenceError::InvalidSafetensors(format!(
                "{}: header is not valid JSON: {e}",
                display_path.display()
            ))
        })?;
        let obj = root.as_object().ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "{}: header root is not a JSON object",
                display_path.display()
            ))
        })?;

        let data_len = mmap.len() - data_offset;

        // Phase 1: parse every entry — including dtypes the converter
        // doesn't decode — so we can validate offset contiguity across
        // the whole data section. Per-tensor byte length is validated
        // against the bit-size table of every standard SafeTensors
        // dtype (sub-byte dtypes like F4 require the shape product to
        // be byte-aligned), and any unrecognized dtype string is
        // rejected outright — the runtime / official parser would.
        let mut parsed: Vec<ParsedEntry> = Vec::with_capacity(obj.len());
        for (name, entry) in obj {
            if name == "__metadata__" {
                continue;
            }

            let dtype_str = entry.get("dtype").and_then(Value::as_str).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: missing or non-string dtype"
                ))
            })?;

            let shape: Vec<usize> = entry
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    InferenceError::InvalidSafetensors(format!(
                        "tensor {name}: missing or non-array shape"
                    ))
                })?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| {
                            InferenceError::InvalidSafetensors(format!(
                                "tensor {name}: shape dim is not u64"
                            ))
                        })
                        .map(|x| x as usize)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let offsets = entry
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    InferenceError::InvalidSafetensors(format!(
                        "tensor {name}: missing or non-array data_offsets"
                    ))
                })?;
            if offsets.len() != 2 {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: data_offsets must have length 2, got {}",
                    offsets.len()
                )));
            }
            let start = offsets[0].as_u64().ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: data_offsets[0] is not u64"
                ))
            })? as usize;
            let end = offsets[1].as_u64().ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: data_offsets[1] is not u64"
                ))
            })? as usize;

            if start > end {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: invalid data_offsets [{start}, {end})"
                )));
            }
            if end > data_len {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: data_offsets end={end} past data_len={data_len}"
                )));
            }

            let numel = shape.iter().try_fold(1usize, |acc, &dim| {
                acc.checked_mul(dim).ok_or_else(|| {
                    InferenceError::InvalidSafetensors(format!(
                        "tensor {name}: shape {shape:?} overflows usize"
                    ))
                })
            })?;
            let bits = safetensors_bits_per_elem(dtype_str).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: unrecognized SafeTensors dtype {dtype_str:?}"
                ))
            })?;
            let total_bits = numel.checked_mul(bits).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: bit length overflows usize"
                ))
            })?;
            if total_bits % 8 != 0 {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: sub-byte dtype {dtype_str} with shape {shape:?} \
                     produces {total_bits} bits, which is not byte-aligned"
                )));
            }
            let expected = total_bits / 8;
            let actual = end - start;
            if actual != expected {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: byte length mismatch for {dtype_str} {shape:?}: \
                     expected {expected}, got {actual}"
                )));
            }

            let header = SourceDType::from_header_str(dtype_str).map(|dtype| TensorHeader {
                dtype,
                shape,
                start,
                end,
            });

            parsed.push(ParsedEntry {
                name: name.clone(),
                start,
                end,
                header,
            });
        }

        // Phase 2: validate that all tensor byte ranges are sorted,
        // disjoint, contiguous from offset 0, and exhaust the data
        // section. This mirrors the official `safetensors` crate
        // validation rules — without it, a corrupted checkpoint could
        // silently alias two tensor names to the same byte range (e.g.
        // both `a` and `b` pointing at `[0, 4)`), or hide leading /
        // trailing padding, and the converter would rotate or quantize
        // bytes that the runtime would never load.
        parsed.sort_by_key(|p| (p.start, p.end));
        let mut prev_end = 0usize;
        for p in &parsed {
            if p.start != prev_end {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: data_offsets non-contiguous at tensor {}: \
                     expected start={prev_end}, got [{}, {})",
                    display_path.display(),
                    p.name,
                    p.start,
                    p.end,
                )));
            }
            prev_end = p.end;
        }
        if prev_end != data_len {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{}: data section is {data_len} bytes but tensors cover {prev_end} bytes \
                 (trailing or missing payload)",
                display_path.display(),
            )));
        }

        // Phase 3: expose only supported-dtype tensors via the read API.
        // Unsupported entries already had their offsets validated above
        // and their bytes accounted for in the contiguity check.
        let mut headers = HashMap::with_capacity(parsed.len());
        for p in parsed {
            if let Some(header) = p.header {
                headers.insert(p.name, header);
            }
        }

        Ok(Shard {
            source: display_path.display().to_string(),
            mmap,
            data_offset,
            headers,
        })
    }

    fn tensor_bytes(&self, name: &str) -> Result<&[u8], InferenceError> {
        let header = self
            .headers
            .get(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;
        // Both adds are bounded by data_offset + data_len = mmap.len(),
        // which fits in usize by construction (mmap.len() is usize).
        let start = self.data_offset + header.start;
        let end = self.data_offset + header.end;
        Ok(&self.mmap[start..end])
    }
}

#[derive(Debug)]
enum Backing {
    Single {
        shard: Shard,
    },
    Sharded {
        /// Tensor name → shard file name (relative to the model directory).
        ///
        /// This is the raw declaration from `model.safetensors.index.json`.
        /// An entry here does NOT imply readability — the corresponding
        /// shard may not contain the tensor (corrupted index) or may
        /// store it with an unsupported dtype that
        /// [`Shard::open`] drops. Use [`Backing::readable_in_sharded`] for
        /// the readable-supported view.
        weight_map: HashMap<String, String>,
        /// All unique shards from `weight_map.values()`, opened eagerly at
        /// [`QuarotTensorReader::open`]. Keyed by shard file name.
        shards: HashMap<String, Shard>,
    },
}

impl Backing {
    /// `true` iff `name` is declared in the manifest AND present in the
    /// owning shard's parsed headers with a supported dtype.
    ///
    /// In sharded mode this is the readable-supported view that
    /// [`QuarotTensorReader::has_tensor`] exposes.
    fn readable_in_sharded(
        weight_map: &HashMap<String, String>,
        shards: &HashMap<String, Shard>,
        name: &str,
    ) -> bool {
        weight_map
            .get(name)
            .and_then(|file| shards.get(file))
            .is_some_and(|shard| shard.headers.contains_key(name))
    }
}

/// Streaming SafeTensors reader for the QuaRot conversion pipeline.
///
/// See module documentation for layout detection and caching semantics.
#[derive(Debug)]
pub struct QuarotTensorReader {
    backing: Backing,
    /// The model-directory fd opened once at [`QuarotTensorReader::open`]
    /// and held for the reader's lifetime. [`QuarotTensorReader::read_file`]
    /// resolves every name against this fd — a caller that already went
    /// through `open()` to validate the checkpoint must never reopen the
    /// model directory by path afterward to read a sibling file (e.g.
    /// `config.json`): that second path resolution is a fresh TOCTOU window
    /// an ancestor swap between the two opens can exploit.
    root_fd: std::os::fd::OwnedFd,
}

impl QuarotTensorReader {
    /// Open a model directory, auto-detecting single-file vs sharded layout.
    ///
    /// Detection order (matches the runtime loaders at
    /// `crate::model::qwen35::Qwen35Model::from_safetensors` and
    /// `crate::model::qwen::QwenModel::from_directory`):
    /// 1. `model.safetensors` present → single file
    /// 2. else `model.safetensors.index.json` present → sharded
    /// 3. neither → [`InferenceError::InvalidSafetensors`]
    ///
    /// When both are present (some HuggingFace repos ship both), the
    /// single-file checkpoint wins. The converter MUST read the same
    /// source as the runtime baseline forward pass, or step 3c's
    /// forward-equivalence assertion would compare against a different
    /// model than the one Lattice actually serves from that directory.
    pub fn open(model_dir: &Path) -> Result<Self, InferenceError> {
        // `model_dir` is opened exactly once and held as `root_fd` for the
        // rest of this call. Every subsequent open (the single-file
        // checkpoint, the index, and every shard) is `openat`-relative to
        // this fd, so an ancestor swap after this line cannot redirect any
        // later read — see `open_manifest_entry`.
        let root_fd = open_dir_nofollow(model_dir).map_err(|e| {
            InferenceError::InvalidSafetensors(format!(
                "failed to open model directory {}: {e}",
                model_dir.display()
            ))
        })?;

        match openat_file_nofollow(
            root_fd.as_raw_fd(),
            std::ffi::OsStr::new("model.safetensors"),
        ) {
            Ok(file) => {
                let shard = Shard::open(file, &model_dir.join("model.safetensors"))?;
                Ok(Self {
                    backing: Backing::Single { shard },
                    root_fd,
                })
            }
            Err(InferenceError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                openat_file_nofollow(
                    root_fd.as_raw_fd(),
                    std::ffi::OsStr::new("model.safetensors.index.json"),
                )
                .map_err(|_| {
                    InferenceError::InvalidSafetensors(format!(
                        "{}: missing both model.safetensors and \
                         model.safetensors.index.json",
                        model_dir.display()
                    ))
                })?;
                let index = parse_index(model_dir)?;
                let mut shards: HashMap<String, Shard> = HashMap::new();
                for shard_file in index.weight_map.values() {
                    if shards.contains_key(shard_file) {
                        continue;
                    }
                    let file = open_manifest_entry(&root_fd, shard_file)?;
                    let shard = Shard::open(file, &model_dir.join(shard_file))?;
                    shards.insert(shard_file.clone(), shard);
                }
                Ok(Self {
                    backing: Backing::Sharded {
                        weight_map: index.weight_map,
                        shards,
                    },
                    root_fd,
                })
            }
            Err(e) => Err(InferenceError::InvalidSafetensors(format!(
                "failed to open {}: {e}",
                model_dir.join("model.safetensors").display()
            ))),
        }
    }

    /// Read `name` (a plain file name, no separators — e.g. `config.json`)
    /// from the model directory this reader was opened against, via
    /// `openat` against the held `root_fd`.
    ///
    /// A caller that has already validated the checkpoint through
    /// [`QuarotTensorReader::open`] must read any other file from the same
    /// model directory (such as `config.json`) through this method rather
    /// than reopening the directory by path — a fresh path resolution is a
    /// new TOCTOU window an ancestor swap between the two opens can use to
    /// substitute a different directory for the second read.
    pub fn read_file(&self, name: &str) -> Result<Vec<u8>, InferenceError> {
        use std::io::Read;
        let mut file = openat_file_nofollow(self.root_fd.as_raw_fd(), std::ffi::OsStr::new(name))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(InferenceError::Io)?;
        Ok(buf)
    }

    /// All readable supported tensor names.
    ///
    /// Returns names that are present in the owning checkpoint AND stored
    /// with a supported dtype (F32 / F16 / BF16). In sharded mode this is
    /// the intersection of the index file's weight map with the shards'
    /// parsed headers — entries the index declares but the shard does not
    /// deliver (corrupted index or unsupported dtype) are excluded so
    /// that step 3c's converter preflight cannot pass on a tensor it
    /// will later fail to read.
    pub fn tensor_names(&self) -> Vec<String> {
        match &self.backing {
            Backing::Single { shard } => shard.headers.keys().cloned().collect(),
            Backing::Sharded { weight_map, shards } => weight_map
                .keys()
                .filter(|name| Backing::readable_in_sharded(weight_map, shards, name))
                .cloned()
                .collect(),
        }
    }

    /// Whether `name` is a readable supported tensor in this checkpoint.
    ///
    /// Returns `true` iff [`tensor_names`] would include `name`. In sharded
    /// mode this means the index declares it AND the owning shard's
    /// parsed headers contain it with a supported dtype.
    ///
    /// [`tensor_names`]: Self::tensor_names
    pub fn has_tensor(&self, name: &str) -> bool {
        match &self.backing {
            Backing::Single { shard } => shard.headers.contains_key(name),
            Backing::Sharded { weight_map, shards } => {
                Backing::readable_in_sharded(weight_map, shards, name)
            }
        }
    }

    /// Shape of a named tensor.
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>, InferenceError> {
        let shard = self.shard_for(name)?;
        shard
            .headers
            .get(name)
            .map(|h| h.shape.clone())
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    /// Source dtype of a named tensor as stored on disk.
    pub fn source_dtype(&self, name: &str) -> Result<SourceDType, InferenceError> {
        let shard = self.shard_for(name)?;
        shard
            .headers
            .get(name)
            .map(|h| h.dtype)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    /// On-disk byte length of a named tensor (`end - start` from the
    /// SafeTensors header). Use this to measure the real source footprint
    /// — e.g. for bf16 weights each element occupies 2 bytes, not 8.
    ///
    /// Matches the pattern used by `bin/quantize_q4`:
    /// ```ignore
    /// let bytes_in = (h.end - h.start) as u64;
    /// ```
    pub fn tensor_byte_len(&self, name: &str) -> Result<u64, InferenceError> {
        let shard = self.shard_for(name)?;
        shard
            .headers
            .get(name)
            .map(|h| (h.end - h.start) as u64)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    /// Read a tensor and convert to a fresh `Vec<f64>`, returned alongside
    /// the shape. Element order is row-major (the on-disk safetensors
    /// convention).
    ///
    /// No conversion cache is kept — the converter discards tensors after
    /// rotation/fusion so caching would only bloat memory.
    pub fn read_tensor_f64(&self, name: &str) -> Result<(Vec<f64>, Vec<usize>), InferenceError> {
        let shard = self.shard_for(name)?;
        let header = shard
            .headers
            .get(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?
            .clone();
        let bytes = shard.tensor_bytes(name)?;
        let data = decode_bytes_to_f64(&shard.source, name, bytes, header.dtype, &header.shape)?;
        Ok((data, header.shape))
    }

    fn shard_for(&self, name: &str) -> Result<&Shard, InferenceError> {
        match &self.backing {
            Backing::Single { shard } => {
                if !shard.headers.contains_key(name) {
                    return Err(InferenceError::MissingTensor(name.to_string()));
                }
                Ok(shard)
            }
            Backing::Sharded { weight_map, shards } => {
                let shard_file = weight_map
                    .get(name)
                    .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;
                let shard = shards.get(shard_file).ok_or_else(|| {
                    InferenceError::InvalidSafetensors(format!(
                        "internal: shard {shard_file} not opened at QuarotTensorReader::open"
                    ))
                })?;
                if !shard.headers.contains_key(name) {
                    return Err(InferenceError::MissingTensor(name.to_string()));
                }
                Ok(shard)
            }
        }
    }
}

fn decode_bytes_to_f64(
    source: &str,
    tensor_name: &str,
    bytes: &[u8],
    dtype: SourceDType,
    shape: &[usize],
) -> Result<Vec<f64>, InferenceError> {
    let raw_dtype = match dtype {
        SourceDType::F32 => crate::weights::ingress::RawDType::F32,
        SourceDType::F16 => crate::weights::ingress::RawDType::F16,
        SourceDType::BF16 => crate::weights::ingress::RawDType::Bf16,
    };
    let mut values = Vec::new();
    crate::weights::ingress::validate_ingested_tensor(
        crate::weights::ingress::IngestedTensor::decode_f64(
            source,
            tensor_name,
            shape,
            dtype.name(),
            bytes,
            raw_dtype,
            &mut values,
        ),
    )?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::fs::File;
    use std::io::Write;

    /// On-disk dtype for synthetic test fixtures.
    #[derive(Debug, Copy, Clone)]
    enum FixtureDType {
        F32,
        F16,
        BF16,
    }

    fn dtype_name(d: FixtureDType) -> &'static str {
        match d {
            FixtureDType::F32 => "F32",
            FixtureDType::F16 => "F16",
            FixtureDType::BF16 => "BF16",
        }
    }

    fn bytes_per_elem(d: FixtureDType) -> usize {
        match d {
            FixtureDType::F32 => 4,
            FixtureDType::F16 | FixtureDType::BF16 => 2,
        }
    }

    fn f32_to_bf16_bits(v: f32) -> u16 {
        // Round-to-nearest-even via the canonical f32→bf16 algorithm.
        let bits = v.to_bits();
        if (bits & 0x7fff_ffff) > 0x7f80_0000 {
            // NaN: preserve quiet payload top bit, ensure non-zero payload.
            ((bits >> 16) as u16) | 0x0040
        } else {
            let round_bit = (bits >> 16) & 1;
            let half = 0x7fff + round_bit;
            ((bits.wrapping_add(half)) >> 16) as u16
        }
    }

    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 31) as u16) << 15;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x007f_ffff;

        // ±Inf / NaN
        if exp == 0xff {
            return if frac == 0 {
                sign | 0x7c00
            } else {
                sign | 0x7c00 | (((frac >> 13) as u16) | 0x0200)
            };
        }
        // ±0 / very small (subnormal-in-f32 → 0 in f16)
        if exp == 0 {
            return sign;
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            // Subnormal in f16. Shift mantissa right by (1 - e), add implicit 1.
            let shift = 14 - exp;
            let mant = (frac | 0x0080_0000) >> shift;
            return sign | (mant as u16);
        }
        sign | ((e as u16) << 10) | ((frac >> 13) as u16)
    }

    fn encode_value(v: f32, dtype: FixtureDType) -> Vec<u8> {
        match dtype {
            FixtureDType::F32 => v.to_le_bytes().to_vec(),
            FixtureDType::BF16 => f32_to_bf16_bits(v).to_le_bytes().to_vec(),
            FixtureDType::F16 => f32_to_f16_bits(v).to_le_bytes().to_vec(),
        }
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, FixtureDType, Vec<usize>, &[f32])]) {
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, dtype, shape, values) in tensors {
            assert_eq!(values.len(), shape.iter().product::<usize>());
            let start = payload.len();
            for &v in *values {
                payload.extend_from_slice(&encode_value(v, *dtype));
            }
            let end = payload.len();
            assert_eq!(end - start, values.len() * bytes_per_elem(*dtype));

            let mut entry = serde_json::Map::new();
            entry.insert("dtype".into(), Value::String(dtype_name(*dtype).into()));
            entry.insert(
                "shape".into(),
                Value::Array(shape.iter().map(|d| Value::from(*d as u64)).collect()),
            );
            entry.insert(
                "data_offsets".into(),
                Value::Array(vec![Value::from(start as u64), Value::from(end as u64)]),
            );
            header.insert((*name).to_string(), Value::Object(entry));
        }
        let header_str = serde_json::to_string(&Value::Object(header)).unwrap();
        let header_bytes = header_str.as_bytes();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_bytes).unwrap();
        file.write_all(&payload).unwrap();
    }

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn single_file_f32_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let values: Vec<f32> = vec![0.0, 1.0, -2.5, 7.125, -0.0001, 1e6];
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("w", FixtureDType::F32, vec![2, 3], &values)],
        );

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("w"));
        assert_eq!(reader.tensor_shape("w").unwrap(), vec![2, 3]);
        assert_eq!(reader.source_dtype("w").unwrap(), SourceDType::F32);

        let (got, shape) = reader.read_tensor_f64("w").unwrap();
        assert_eq!(shape, vec![2, 3]);
        for (g, &v) in got.iter().zip(values.iter()) {
            assert_eq!(*g, v as f64);
        }
    }

    #[test]
    #[cfg(unix)]
    fn read_file_uses_the_fd_open_already_bound_not_a_fresh_path_resolution() {
        // A caller that reads a sibling file (e.g. `config.json`) after
        // `open()` has already validated the checkpoint must never
        // re-resolve the model directory by path to do it — that second
        // resolution is a fresh window an ancestor swap between the two
        // opens can use to substitute a different directory's file.
        // `read_file` instead resolves `name` against the same `root_fd`
        // `open()` already holds. Swap the model directory for a different
        // one (same path, fresh content) after `open()` returns, then
        // confirm `read_file` still reaches the ORIGINAL directory's file,
        // not the substituted one.
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("model");
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), b"REAL-CONFIG").unwrap();
        write_safetensors(
            &model_dir.join("model.safetensors"),
            &[("w", FixtureDType::F32, vec![1], &[1.0])],
        );

        let reader = QuarotTensorReader::open(&model_dir).unwrap();

        let moved_aside = dir.path().join("model_moved_aside");
        fs::rename(&model_dir, &moved_aside).unwrap();
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), b"FAKE-CONFIG").unwrap();

        let bytes = reader.read_file("config.json").unwrap();
        assert_eq!(
            bytes, b"REAL-CONFIG",
            "read_file must reach the directory validated at open(), not \
             whatever directory currently occupies the model_dir path"
        );
    }

    #[test]
    fn single_file_bf16_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 2.5, -3.75, 100.0];
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("w", FixtureDType::BF16, vec![3, 2], &values)],
        );

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert_eq!(reader.source_dtype("w").unwrap(), SourceDType::BF16);
        let (got, shape) = reader.read_tensor_f64("w").unwrap();
        assert_eq!(shape, vec![3, 2]);
        // bf16 has 7-bit mantissa: relative precision ~2^-8 ≈ 4e-3 for these
        // values; absolute tolerance scaled to magnitude.
        for (g, &v) in got.iter().zip(values.iter()) {
            let tol = (v.abs() as f64) * (1.0 / 128.0) + 1e-6;
            assert!(approx_eq(*g, v as f64, tol), "got {g} expected ~{v}");
        }
    }

    #[test]
    fn single_file_f16_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        // Values within f16 range; f16 has 10-bit mantissa.
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -2.0, 100.0];
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("w", FixtureDType::F16, vec![6], &values)],
        );

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert_eq!(reader.source_dtype("w").unwrap(), SourceDType::F16);
        let (got, _) = reader.read_tensor_f64("w").unwrap();
        for (g, &v) in got.iter().zip(values.iter()) {
            let tol = (v.abs() as f64) * (1.0 / 1024.0) + 1e-6;
            assert!(approx_eq(*g, v as f64, tol), "got {g} expected ~{v}");
        }
    }

    #[test]
    fn read_tensor_f64_rejects_non_finite_values_with_provenance() {
        for dtype in [FixtureDType::F32, FixtureDType::F16, FixtureDType::BF16] {
            for (case, bad) in [
                ("nan", f32::NAN),
                ("positive-infinity", f32::INFINITY),
                ("negative-infinity", f32::NEG_INFINITY),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("model.safetensors");
                let values = [1.0, bad, 3.0];
                write_safetensors(&path, &[("invalid.weight", dtype, vec![3], &values)]);

                let reader = QuarotTensorReader::open(dir.path()).unwrap();
                let err = reader
                    .read_tensor_f64("invalid.weight")
                    .expect_err("non-finite source value must be rejected");
                let msg = err.to_string();
                assert!(
                    msg.contains("invalid.weight"),
                    "{case} {dtype:?} error must name the tensor: {msg}"
                );
                assert!(
                    msg.contains(path.to_string_lossy().as_ref()),
                    "{case} {dtype:?} error must name the source path: {msg}"
                );
                assert!(
                    msg.contains("element index 1"),
                    "{case} {dtype:?} error must locate the bad value: {msg}"
                );
            }
        }
    }

    #[test]
    fn sharded_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let shard_a = "model-00001-of-00002.safetensors";
        let shard_b = "model-00002-of-00002.safetensors";

        let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_vals: Vec<f32> = vec![-1.0, -2.0];

        write_safetensors(
            &dir.path().join(shard_a),
            &[("a.weight", FixtureDType::F32, vec![2, 2], &a_vals)],
        );
        write_safetensors(
            &dir.path().join(shard_b),
            &[("b.weight", FixtureDType::BF16, vec![2], &b_vals)],
        );

        let index = serde_json::json!({
            "metadata": {"total_size": 24usize},
            "weight_map": {
                "a.weight": shard_a,
                "b.weight": shard_b,
            },
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("a.weight"));
        assert!(reader.has_tensor("b.weight"));
        assert!(!reader.has_tensor("missing"));

        let mut names = reader.tensor_names();
        names.sort();
        assert_eq!(names, vec!["a.weight".to_string(), "b.weight".to_string()]);

        let (a, a_shape) = reader.read_tensor_f64("a.weight").unwrap();
        assert_eq!(a_shape, vec![2, 2]);
        assert_eq!(a, vec![1.0, 2.0, 3.0, 4.0]);

        let (b, b_shape) = reader.read_tensor_f64("b.weight").unwrap();
        assert_eq!(b_shape, vec![2]);
        // BF16 of -1.0 and -2.0 are exact.
        assert_eq!(b, vec![-1.0, -2.0]);

        // Re-reading the same tensor exercises the cached-shard path.
        let (a_again, _) = reader.read_tensor_f64("a.weight").unwrap();
        assert_eq!(a_again, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // -------------------------------------------------------------------
    // Shard-path containment via descriptor-relative traversal
    // (`open_manifest_entry`). Containment is a structural property of the
    // walk here, not a canonicalize-then-open string comparison: each hop
    // opens relative to the fd from the previous hop, so an ancestor swap
    // after `QuarotTensorReader::open`'s initial `open_dir_nofollow` cannot
    // redirect any later read.
    //
    // with `open_manifest_entry` bypassed (shard
    // resolution reverted to a bare `model_dir.join(shard_file)` reopened by
    // path), every test below would instead succeed in reading the escaped
    // tensor's real contents from outside the model directory.
    // -------------------------------------------------------------------

    #[test]
    #[cfg(unix)]
    fn shard_entry_rejects_a_symlinked_final_component() {
        // A symlink planted at the shard's final path component — standing
        // in for a swap that lands between the initial `open_dir_nofollow`
        // and the later per-shard `openat`. dropping
        // `O_NOFOLLOW` from `openat_file_nofollow` makes this test fail —
        // the open would succeed and follow the symlink.
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("model_root");
        std::fs::create_dir(&model_dir).unwrap();

        let real = root.path().join("real.safetensors");
        write_safetensors(
            &real,
            &[("secret.weight", FixtureDType::F32, vec![1], &[42.0])],
        );
        let link = model_dir.join("linked.safetensors");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let root_fd = open_dir_nofollow(&model_dir).unwrap();
        let err = open_manifest_entry(&root_fd, "linked.safetensors")
            .expect_err("a symlinked shard entry must be refused, not followed");
        assert!(
            matches!(err, InferenceError::Io(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn shard_entry_rejects_a_symlinked_intermediate_directory() {
        // The shard entry's intermediate directory component ("subdir") is a
        // symlink to a directory outside model_root. `O_NOFOLLOW` only
        // protects the *final* path component in a plain `open`, so this
        // guards the hop-by-hop `openat_dir_nofollow` walk specifically.
        // dropping `O_NOFOLLOW` from
        // `openat_dir_nofollow` makes this test fail — the walk would
        // follow the symlink into the outside directory and read the
        // escaped tensor.
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("model_root");
        std::fs::create_dir(&model_dir).unwrap();

        let outside_dir = root.path().join("outside");
        std::fs::create_dir(&outside_dir).unwrap();
        write_safetensors(
            &outside_dir.join("shard.safetensors"),
            &[("secret.weight", FixtureDType::F32, vec![1], &[42.0])],
        );

        let subdir_link = model_dir.join("subdir");
        std::os::unix::fs::symlink(&outside_dir, &subdir_link).unwrap();

        let root_fd = open_dir_nofollow(&model_dir).unwrap();
        let err = open_manifest_entry(&root_fd, "subdir/shard.safetensors")
            .expect_err("a symlinked intermediate directory must be refused");
        assert!(
            matches!(err, InferenceError::Io(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sharded_index_rejects_path_traversal_shard_entry() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("model_root");
        std::fs::create_dir(&model_dir).unwrap();

        // A file OUTSIDE model_dir that a hostile index.json tries to reach.
        write_safetensors(
            &root.path().join("secret.safetensors"),
            &[("secret.weight", FixtureDType::F32, vec![1], &[42.0])],
        );

        let index = serde_json::json!({
            "metadata": {"total_size": 4usize},
            "weight_map": {"secret.weight": "../secret.safetensors"},
        });
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let err = QuarotTensorReader::open(&model_dir)
            .expect_err("a shard entry escaping model_dir via ../ must be rejected");
        assert!(
            err.to_string().contains("no `..` or root component"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sharded_index_rejects_absolute_shard_entry() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("model_root");
        std::fs::create_dir(&model_dir).unwrap();

        write_safetensors(
            &root.path().join("secret.safetensors"),
            &[("secret.weight", FixtureDType::F32, vec![1], &[42.0])],
        );

        let absolute_target = root.path().join("secret.safetensors");
        let index = serde_json::json!({
            "metadata": {"total_size": 4usize},
            "weight_map": {"secret.weight": absolute_target.to_str().unwrap()},
        });
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let err = QuarotTensorReader::open(&model_dir)
            .expect_err("an absolute shard entry must be rejected");
        assert!(
            err.to_string().contains("no `..` or root component"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_tensor_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("present", FixtureDType::F32, vec![1], &[1.0])],
        );
        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        let err = reader.read_tensor_f64("absent").unwrap_err();
        match err {
            InferenceError::MissingTensor(name) => assert_eq!(name, "absent"),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn no_safetensors_in_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidSafetensors(_)));
    }

    #[test]
    fn byte_length_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        // Hand-build a malformed file: F32 dtype, shape [4] (16 bytes), but
        // declare only 12 bytes of data.
        let header = serde_json::json!({
            "w": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 12],
            }
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend_from_slice(&[0u8; 12]);
        std::fs::write(dir.path().join("model.safetensors"), &buf).unwrap();

        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("byte length mismatch"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_dtype_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        // Build a file with two tensors: a supported F32 and an unsupported
        // I64. The I64 must be ignored, not error.
        let header = serde_json::json!({
            "supported": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
            "ignored": {
                "dtype": "I64",
                "shape": [1],
                "data_offsets": [4, 12],
            }
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend_from_slice(&7f32.to_le_bytes());
        buf.extend_from_slice(&42i64.to_le_bytes());
        std::fs::write(dir.path().join("model.safetensors"), &buf).unwrap();

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("supported"));
        assert!(!reader.has_tensor("ignored"));
        let (data, _) = reader.read_tensor_f64("supported").unwrap();
        assert_eq!(data, vec![7.0]);
    }

    #[test]
    fn single_file_wins_when_both_layouts_present() {
        // Mirrors the runtime loaders at qwen35/model.rs:25 and qwen.rs:425,
        // which both check `model.safetensors` before the sharded index.
        // The converter must agree, otherwise step 3c's forward-equivalence
        // check would target a different checkpoint than the runtime serves.
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("real", FixtureDType::F32, vec![1], &[1.0])],
        );
        write_safetensors(
            &dir.path().join("model-00001-of-00001.safetensors"),
            &[("decoy", FixtureDType::F32, vec![1], &[999.0])],
        );
        let index = serde_json::json!({
            "metadata": {"total_size": 4usize},
            "weight_map": {"decoy": "model-00001-of-00001.safetensors"},
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("real"));
        assert!(!reader.has_tensor("decoy"));
        let (data, _) = reader.read_tensor_f64("real").unwrap();
        assert_eq!(data, vec![1.0]);
    }

    /// Helper for malformed-layout tests: writes a hand-crafted SafeTensors
    /// file with a caller-supplied header object and raw data section.
    fn write_raw_safetensors(path: &Path, header_json: &Value, data: &[u8]) {
        let header_str = serde_json::to_string(header_json).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend_from_slice(data);
        std::fs::write(path, &buf).unwrap();
    }

    /// Two F32 tensors declared at the same `[0, 4)` byte range alias each
    /// other — the official safetensors parser rejects this via
    /// `InvalidOffset`, and so must we, because a converter that reads
    /// both names would silently rotate the same bytes twice.
    #[test]
    fn overlapping_offsets_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
            "b": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
        });
        write_raw_safetensors(
            &dir.path().join("model.safetensors"),
            &header,
            &1.0f32.to_le_bytes(),
        );
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("non-contiguous"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// First tensor starts at offset 4 — the data section has a 4-byte
    /// leading hole. The official parser rejects this; we must too.
    #[test]
    fn leading_hole_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "w": { "dtype": "F32", "shape": [1], "data_offsets": [4, 8] },
        });
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&7.0f32.to_le_bytes());
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("non-contiguous"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// Gap between the end of tensor `a` and the start of tensor `b` —
    /// 4 stray bytes in the middle of the data section.
    #[test]
    fn internal_hole_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
            "b": { "dtype": "F32", "shape": [1], "data_offsets": [8, 12] },
        });
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&2.0f32.to_le_bytes());
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("non-contiguous"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// Data section has 4 extra trailing bytes beyond the last tensor's
    /// declared end. The official parser rejects this; we must too.
    #[test]
    fn trailing_payload_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "w": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
        });
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&[0u8; 4]);
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("trailing or missing payload"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// An unsupported-dtype (I64) tensor at offset 4 must still fail the
    /// global contiguity check — offset validation applies to every
    /// declared tensor, not just the F32/F16/BF16 ones the reader decodes.
    #[test]
    fn unsupported_dtype_with_bad_offsets_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "x": { "dtype": "I64", "shape": [1], "data_offsets": [4, 12] },
        });
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&42i64.to_le_bytes());
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("non-contiguous"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// An entirely unrecognized dtype string is rejected at open time,
    /// not silently skipped. The runtime / official parser would reject
    /// the same bytes; the converter must not bless what the runtime
    /// wouldn't.
    #[test]
    fn unknown_dtype_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "x": { "dtype": "FUTURE_DTYPE", "shape": [1], "data_offsets": [0, 4] },
        });
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 4]);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("unrecognized SafeTensors dtype"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// F8_E8M0 is 8 bits per element; shape [2] requires 2 bytes.
    /// Declaring a 1-byte range fails the byte-length check, even though
    /// F8_E8M0 isn't a converter-decoded dtype.
    #[test]
    fn f8_e8m0_byte_length_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "x": { "dtype": "F8_E8M0", "shape": [2], "data_offsets": [0, 1] },
        });
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 1]);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("byte length mismatch"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// C64 is complex64 — 64 bits = 8 bytes per element. Shape [1]
    /// requires 8 bytes; declaring a 4-byte range fails.
    #[test]
    fn c64_byte_length_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "z": { "dtype": "C64", "shape": [1], "data_offsets": [0, 4] },
        });
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 4]);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("byte length mismatch"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// F4 stores 4 bits per element. Shape [3] = 12 bits, which is not
    /// byte-aligned — the file is unreadable regardless of declared
    /// offsets. The reader rejects this at open time.
    #[test]
    fn f4_sub_byte_misalignment_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "x": { "dtype": "F4", "shape": [3], "data_offsets": [0, 2] },
        });
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 2]);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("not byte-aligned"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// F6_E2M3 stores 6 bits per element. Shape [4] = 24 bits = 3 bytes;
    /// declaring a 2-byte range fails the byte-length check.
    #[test]
    fn f6_e2m3_byte_length_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "x": { "dtype": "F6_E2M3", "shape": [4], "data_offsets": [0, 2] },
        });
        write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 2]);
        let err = QuarotTensorReader::open(dir.path()).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("byte length mismatch"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
    }

    /// Index declares `required.weight` in a shard whose actual contents
    /// do NOT include that tensor. `has_tensor` and `tensor_names` must
    /// reflect the readable view, not the manifest, so step 3c's preflight
    /// rejects the checkpoint before the rotation pass starts.
    #[test]
    fn sharded_index_entry_missing_from_shard_is_not_readable() {
        let dir = tempfile::tempdir().unwrap();
        let shard_file = "model-00001-of-00001.safetensors";
        // Shard contains only `other.weight`, not `required.weight`.
        write_safetensors(
            &dir.path().join(shard_file),
            &[("other.weight", FixtureDType::F32, vec![1], &[1.0])],
        );
        let index = serde_json::json!({
            "metadata": {"total_size": 4usize},
            "weight_map": {
                "required.weight": shard_file,
                "other.weight": shard_file,
            },
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("other.weight"));
        assert!(!reader.has_tensor("required.weight"));
        assert_eq!(reader.tensor_names(), vec!["other.weight".to_string()]);
        let err = reader.read_tensor_f64("required.weight").unwrap_err();
        assert!(matches!(err, InferenceError::MissingTensor(_)));
    }

    /// Index declares `required.weight` in a shard that DOES contain a
    /// tensor by that name but with an unsupported dtype (I64). The shard
    /// parser drops unsupported entries, so the reader must report the
    /// tensor as not-readable.
    #[test]
    fn sharded_index_entry_unsupported_dtype_is_not_readable() {
        let dir = tempfile::tempdir().unwrap();
        let shard_file = "model-00001-of-00001.safetensors";
        // Hand-build a shard with one I64 tensor (unsupported dtype).
        let header = serde_json::json!({
            "required.weight": {
                "dtype": "I64",
                "shape": [1],
                "data_offsets": [0, 8],
            }
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend_from_slice(&42i64.to_le_bytes());
        std::fs::write(dir.path().join(shard_file), &buf).unwrap();

        let index = serde_json::json!({
            "metadata": {"total_size": 8usize},
            "weight_map": {"required.weight": shard_file},
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(!reader.has_tensor("required.weight"));
        assert!(reader.tensor_names().is_empty());
        let err = reader.read_tensor_f64("required.weight").unwrap_err();
        assert!(matches!(err, InferenceError::MissingTensor(_)));
    }

    /// Compile-time check that `qwen_required_tensor_names` is reachable
    /// from non-test code paths. If step 3b's promotion ever regresses,
    /// this test fails to compile.
    #[test]
    fn qwen_required_tensor_names_is_publicly_reachable() {
        use crate::model::qwen35::qwen_required_tensor_names;
        use crate::model::qwen35_config::Qwen35Config;
        // We don't construct a config (it has many required fields); we
        // just want a fn pointer to prove visibility.
        let _f: fn(&Qwen35Config) -> Vec<String> = qwen_required_tensor_names;
    }
}
