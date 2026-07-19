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

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

use crate::error::InferenceError;
use crate::model::qwen35::qwen_layer_tensor_prefix;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::plan::{OnlineRotationSpec, OnlineTransformSite};
use crate::weights::f32_weights::{contained_shard_path, parse_index};

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
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "I16" | "U16" | "F16" | "BF16" => Some(16),
        "I32" | "U32" | "F32" => Some(32),
        "I64" | "U64" | "F64" | "C64" => Some(64),
        _ => None,
    }
}

impl Shard {
    fn open(path: &Path) -> Result<Self, InferenceError> {
        let file = File::open(path).map_err(|e| {
            InferenceError::InvalidSafetensors(format!("failed to open {}: {e}", path.display()))
        })?;
        // SAFETY: the file is opened read-only; the returned Mmap owns the
        // mapping. The File handle can be dropped immediately after — the OS
        // keeps the fd alive through the map. Standard memmap2 usage.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            InferenceError::InvalidSafetensors(format!("failed to mmap {}: {e}", path.display()))
        })?;

        if mmap.len() < 8 {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{}: file too small to contain a safetensors header",
                path.display()
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
                path.display(),
                data_offset,
                mmap.len()
            )));
        }

        let header_str = std::str::from_utf8(&mmap[8..data_offset]).map_err(|e| {
            InferenceError::InvalidSafetensors(format!(
                "{}: header is not valid UTF-8: {e}",
                path.display()
            ))
        })?;
        let root: Value = serde_json::from_str(header_str).map_err(|e| {
            InferenceError::InvalidSafetensors(format!(
                "{}: header is not valid JSON: {e}",
                path.display()
            ))
        })?;
        let obj = root.as_object().ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "{}: header root is not a JSON object",
                path.display()
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
                    path.display(),
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
                path.display(),
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

/// Version tag for the QuaRot artifact index (`quantize_index.json`).
/// This is wired into the actual manifest schema:
/// `crate::quant::quarot::convert::QuantizeIndex` carries an optional
/// `online: Option<OnlineArtifactDescriptor>` field, and
/// `read_quarot_seed_from_index` validates it (via
/// [`OnlineArtifactDescriptor::validate`]) whenever it is present. A
/// *structurally valid* `V1Online` descriptor is rejected there too (not
/// merely validated-then-discarded), because no forward path executes R3/R4
/// online rotations at runtime yet — see that function's doc comment for the
/// full fail-closed contract.
///
/// `V0Residual` is every artifact produced by today's `convert_quarot_qwen35`
/// (ADR-044/051): offline residual-stream rotation only, symmetric-only Q4,
/// no R3/R4 metadata. `V1Online` pairs that offline rotation with one or
/// more online (runtime) R3/R4 rotations and requires every Q4 tensor the
/// online rotation touches to explicitly declare asymmetric mode.
///
/// **Hard rule (design doc §E.1(a)):** a `V0Residual` artifact must never be
/// interpreted as online-capable. [`OnlineArtifactDescriptor::validate`]
/// enforces this by refusing a `V0Residual` descriptor that carries any
/// [`OnlineRotationSpec`] — that combination cannot occur from a real v0
/// artifact and is evidence of a corrupted or hand-edited index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ArtifactVersion {
    #[serde(rename = "v0-residual")]
    V0Residual,
    #[serde(rename = "v1-online-r3r4")]
    V1Online,
}

/// Artifact-level descriptor pairing an [`ArtifactVersion`] with its online
/// rotation recipe and per-tensor asymmetric-Q4 declarations. This type
/// lives alongside the streaming reader rather than in `convert.rs` because
/// `convert.rs` owns the wire-format struct (`QuantizeIndex`) while
/// validation semantics live here; `QuantizeIndex`
/// embeds this type directly (`online: Option<OnlineArtifactDescriptor>`)
/// and `read_quarot_seed_from_index` calls [`Self::validate`] on it when
/// present — this is no longer schema-only, it is consulted at load time
/// (and a valid `V1Online` descriptor causes the load to be rejected
/// outright, since no forward path executes its recipe yet).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OnlineArtifactDescriptor {
    pub version: ArtifactVersion,
    /// v1 contract: at most one spec per online transform site (per
    /// [`super::plan::RotationId::online_transform_site`]) — a second spec
    /// targeting the same site is rejected by [`Self::validate`] rather
    /// than left for a future runtime to disambiguate.
    pub online_rotations: Vec<OnlineRotationSpec>,
    /// Names of every Q4 tensor that is affected by an online rotation
    /// (i.e., every tensor an entry in `online_rotations` counter-rotates)
    /// AND is stored asymmetric. Design doc §E.1(b): "an online artifact
    /// declares asymmetric Q4 mode explicitly for every affected tensor."
    pub asymmetric_tensor_names: Vec<String>,
}

impl OnlineArtifactDescriptor {
    /// Upper bound on each caller-supplied vector (`online_rotations` and
    /// `asymmetric_tensor_names`). A descriptor is publicly deserializable, so
    /// [`Self::validate`] caps both vectors before any per-entry work: a real
    /// recipe declares a handful of rotations and at most a few tensors per
    /// layer, so this ceiling sits far above any legitimate artifact while
    /// keeping validation cost strictly bounded regardless of input.
    const MAX_VALIDATED_ENTRIES: usize = 4096;

    /// Derive the exact set of per-layer tensor names this descriptor's
    /// online rotations counter-rotate, from the recipe (`online_rotations`)
    /// and `cfg` alone. This is the SINGLE internal representation of "what
    /// this artifact affects" — [`Self::validate`] requires
    /// `asymmetric_tensor_names` to equal this set exactly. This collapses
    /// the prior three-representation contract — recipe-derived suffixes,
    /// declared `asymmetric_tensor_names`, and a caller-supplied affected
    /// slice — down to this one derived set plus the single external
    /// declaration.
    ///
    /// Each entry pairs the owning spec's [`super::plan::RotationId`] and
    /// layer index with the derived tensor name, so callers can report
    /// which spec/layer a missing or unexpected name belongs to.
    fn derive_expected_tensor_names(
        &self,
        cfg: &Qwen35Config,
    ) -> Vec<(super::plan::RotationId, usize, String)> {
        let mut expected = Vec::new();
        for spec in &self.online_rotations {
            let Some(site) = spec.id.online_transform_site() else {
                continue;
            };
            let suffix = site.weight_tensor_suffix();
            let layers: Vec<usize> = match &spec.layer_scope {
                Some(layers) => layers.clone(),
                None => (0..cfg.num_hidden_layers).collect(),
            };
            for idx in layers {
                expected.push((
                    spec.id,
                    idx,
                    format!("{}.{suffix}", qwen_layer_tensor_prefix(idx)),
                ));
            }
        }
        expected
    }

    /// Cross-check that a converter's actually-transformed tensor names
    /// exactly match this descriptor's derived affected-tensor set (see
    /// [`Self::derive_expected_tensor_names`]). Narrow entry point for a
    /// future converter that wants to verify what it actually transformed
    /// against the recipe — the core [`Self::validate`] path does not take
    /// this input and does not call this method.
    ///
    /// `pub(crate)`, not `pub`: no caller (in-crate or cross-crate) exists
    /// yet — this is a hook for a future converter, not a stable external
    /// API. Widen to `pub` if/when that converter lands and needs to call
    /// it from outside this crate. `allow(dead_code)` outside `test` cfg
    /// for the same reason: today only this module's tests call it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn verify_affected(
        &self,
        cfg: &Qwen35Config,
        transformed_tensor_names: &[&str],
    ) -> Result<(), InferenceError> {
        let expected = self.derive_expected_tensor_names(cfg);
        for (id, idx, name) in &expected {
            if !transformed_tensor_names.contains(&name.as_str()) {
                return Err(InferenceError::Inference(format!(
                    "OnlineArtifactDescriptor::verify_affected: {id:?} recipe \
                     is scoped to layer {idx}, but the converter's \
                     transformed-tensor list does not include the expected \
                     tensor {name:?}"
                )));
            }
        }
        for name in transformed_tensor_names {
            if !expected.iter().any(|(_, _, e)| e == name) {
                return Err(InferenceError::Inference(format!(
                    "OnlineArtifactDescriptor::verify_affected: converter \
                     reports transforming {name:?}, which this descriptor's \
                     online rotations do not counter-rotate"
                )));
            }
        }
        Ok(())
    }

    /// Validate internal consistency, including that `asymmetric_tensor_names`
    /// equals exactly the tensor set this descriptor's online rotations
    /// derive from `spec` + `cfg` (see `Self::derive_expected_tensor_names`).
    ///
    /// Refuses (does not warn-and-continue) on:
    /// - `V0Residual` carrying any online rotation metadata — a v0 artifact
    ///   is never online-capable, so a non-empty `online_rotations` on a
    ///   `V0Residual` descriptor is a corrupted/hand-edited index.
    /// - `V1Online` declaring zero online rotations — an online artifact
    ///   with no R3/R4 recipe is a contradiction, not a valid degenerate
    ///   case (use `V0Residual` for that).
    /// - `V1Online` with any rotation spec that fails its own internal
    ///   invariants (see [`OnlineRotationSpec::validate`]) — e.g. wrong
    ///   `side`, non-power-of-two `block_size`, or a `layer_scope` that
    ///   contradicts its `RotationId`'s contract. Every `OnlineRotationSpec`
    ///   field is publicly constructible, so a directly-built malformed spec
    ///   (bypassing the `r3_full_attention`/`r4_dense_mlp` constructors)
    ///   must be caught here rather than passing descriptor validation
    ///   silently. `cfg` is threaded
    ///   through to each spec as `Some(cfg)` so the config-aware
    ///   divisibility/full-attention checks always run for `V1Online` (see
    ///   below: `cfg` is mandatory at the descriptor level for that
    ///   variant). Passing `cfg: None` to *this* method is valid only for
    ///   an empty `V0Residual` descriptor: `V1Online` unconditionally
    ///   rejects `cfg: None` below, and `V0Residual` never carries a spec to
    ///   thread `cfg` into in the first place.
    /// - `V1Online` with more than one spec in `online_rotations` targeting
    ///   the same [`OnlineTransformSite`] (e.g. two `AttentionOutputR3`
    ///   entries with different seeds) — the v1 recipe format allows at
    ///   most one spec per runtime site, so a duplicate is rejected rather
    ///   than left for a future runtime to decide whether to apply both,
    ///   pick one, or merge them.
    /// - `V1Online` with a nonempty rotation recipe but an empty
    ///   `asymmetric_tensor_names` list — self-contradictory: a
    ///   rotation-bearing recipe always counter-rotates at least one stored
    ///   tensor, so an empty declaration list cannot be satisfied.
    /// - `V1Online` with any `asymmetric_tensor_names` entry that is not in
    ///   the exact set `Self::derive_expected_tensor_names` derives from
    ///   this descriptor's `online_rotations` + `cfg` — an unrelated or
    ///   out-of-scope declaration (e.g. `"unrelated"`, or a real tensor name
    ///   from a layer outside the recipe's scope) must not satisfy the
    ///   fail-closed requirement just by resembling a real tensor.
    /// - `V1Online` where a scoped rotation's per-layer tensor coverage is
    ///   incomplete: matching a tensor
    ///   name by *suffix* alone (e.g. `self_attn.o_proj.weight`) is not
    ///   sufficient — an R3 recipe scoped to layers `[3,7,11,15,19,23]`
    ///   must not validate against a declaration set that only contains
    ///   layer 0's tensor, even though that name has the right suffix.
    ///   `asymmetric_tensor_names` is the **sole externally-supplied**
    ///   representation of the affected-tensor contract (it collapses the
    ///   prior recipe-suffix / declaration / caller-supplied-slice triple
    ///   representation down to this one) and must equal
    ///   `Self::derive_expected_tensor_names(cfg)` exactly — no missing
    ///   entries, no extras. A future converter that wants to
    ///   cross-check what it actually transformed against this contract
    ///   should use `Self::verify_affected`, which `validate` itself does
    ///   not call.
    /// - `V1Online` called with `cfg: None`: a `None`-scoped (R4)
    ///   rotation's full layer set is
    ///   unknowable without a config, so a config-free call previously
    ///   *skipped* per-layer completeness checking for R4 instead of
    ///   refusing — letting a caller silently select weaker semantics by
    ///   omitting `cfg`. `cfg` is now mandatory for `V1Online`; this method
    ///   refuses immediately rather than downgrading to a partial check.
    ///   `V0Residual` never claims per-layer completeness, so it stays
    ///   `cfg`-free.
    /// - `V0Residual` with any nonempty `asymmetric_tensor_names`: the V0
    ///   contract is symmetric-only Q4
    ///   with no R3/R4 metadata, so a V0 descriptor carrying asymmetric
    ///   declarations is evidence of a corrupted or hand-edited index, same
    ///   as a V0 descriptor carrying online rotations.
    pub fn validate(&self, cfg: Option<&Qwen35Config>) -> Result<(), InferenceError> {
        if self.online_rotations.len() > Self::MAX_VALIDATED_ENTRIES {
            return Err(InferenceError::Inference(format!(
                "OnlineArtifactDescriptor: online_rotations declares {} entries, \
                 exceeding the maximum of {} — a descriptor this large is rejected \
                 before validation to keep work bounded over untrusted input",
                self.online_rotations.len(),
                Self::MAX_VALIDATED_ENTRIES
            )));
        }
        if self.asymmetric_tensor_names.len() > Self::MAX_VALIDATED_ENTRIES {
            return Err(InferenceError::Inference(format!(
                "OnlineArtifactDescriptor: asymmetric_tensor_names declares {} entries, \
                 exceeding the maximum of {} — a descriptor this large is rejected \
                 before validation to keep work bounded over untrusted input",
                self.asymmetric_tensor_names.len(),
                Self::MAX_VALIDATED_ENTRIES
            )));
        }
        match self.version {
            ArtifactVersion::V0Residual => {
                if !self.online_rotations.is_empty() {
                    return Err(InferenceError::Inference(
                        "OnlineArtifactDescriptor: a V0Residual artifact must not \
                         carry online rotation metadata — a v0 residual artifact \
                         is never interpreted as online-capable"
                            .to_string(),
                    ));
                }
                if !self.asymmetric_tensor_names.is_empty() {
                    return Err(InferenceError::Inference(
                        "OnlineArtifactDescriptor: a V0Residual artifact must not \
                         declare any asymmetric_tensor_names — the V0 contract is \
                         symmetric-only Q4 with no R3/R4 metadata"
                            .to_string(),
                    ));
                }
            }
            ArtifactVersion::V1Online => {
                let cfg = cfg.ok_or_else(|| {
                    InferenceError::Inference(
                        "OnlineArtifactDescriptor: V1 online validation requires \
                         model config — a config-free call cannot prove per-layer \
                         tensor coverage is complete (an R4 recipe's \
                         layer_scope=None 'every layer' claim is unverifiable \
                         without knowing how many layers the model has), so \
                         cfg=None is refused rather than silently downgrading to \
                         a weaker, incomplete validation path"
                            .to_string(),
                    )
                })?;
                if self.online_rotations.is_empty() {
                    return Err(InferenceError::Inference(
                        "OnlineArtifactDescriptor: a V1Online artifact must declare \
                         at least one R3/R4 online rotation"
                            .to_string(),
                    ));
                }
                for spec in &self.online_rotations {
                    spec.validate(Some(cfg))?;
                }
                // Canonical-form contract (v1): at most one spec per online
                // transform site. Two specs targeting the same runtime site
                // (e.g. two `AttentionOutputR3` entries with different
                // seeds) would leave the future runtime to invent whether
                // to apply both, pick one, or merge them — reject the
                // second spec outright rather than let that ambiguity into
                // the artifact.
                let mut seen_sites: Vec<OnlineTransformSite> = Vec::new();
                for spec in &self.online_rotations {
                    let Some(site) = spec.id.online_transform_site() else {
                        continue;
                    };
                    if seen_sites.contains(&site) {
                        return Err(InferenceError::Inference(format!(
                            "OnlineArtifactDescriptor: more than one online \
                             rotation spec targets the same runtime site \
                             {site:?} — a V1Online artifact must declare at \
                             most one spec per online transform site"
                        )));
                    }
                    seen_sites.push(site);
                }
                // Fail-closed on an empty declaration list: a nonempty
                // rotation recipe always counter-rotates at least one
                // stored tensor, so a v1 descriptor with rotations but zero
                // asymmetric declarations is self-contradictory.
                if self.asymmetric_tensor_names.is_empty() {
                    return Err(InferenceError::Inference(
                        "OnlineArtifactDescriptor: a V1Online artifact with online \
                         rotations must declare asymmetric Q4 mode for the tensors \
                         those rotations affect; an empty asymmetric declaration \
                         list is self-contradictory"
                            .to_string(),
                    ));
                }
                // Single-representation contract: `asymmetric_tensor_names`
                // must equal the recipe-derived expected set exactly, derived
                // once here via `derive_expected_tensor_names` rather than
                // re-collected from any caller-supplied slice. The comparison
                // runs through hash sets so it stays linear in the number of
                // declared and expected names — a descriptor is publicly
                // constructible and deserializable, so a nested-scan form
                // would be quadratic in externally-supplied input — and
                // duplicate declarations are rejected up front so the declared
                // list is a true set. Missing-direction first (names the
                // recipe requires but the descriptor doesn't declare), then
                // extra-direction (declared names the recipe doesn't recognize,
                // matched against the full derived name, not only a suffix).
                let expected = self.derive_expected_tensor_names(cfg);
                let mut declared_set: HashSet<&str> =
                    HashSet::with_capacity(self.asymmetric_tensor_names.len());
                for declared in &self.asymmetric_tensor_names {
                    if !declared_set.insert(declared.as_str()) {
                        return Err(InferenceError::Inference(format!(
                            "OnlineArtifactDescriptor: asymmetric_tensor_names \
                             declares {declared:?} more than once — each affected \
                             tensor must be declared exactly once"
                        )));
                    }
                }
                for (id, idx, expected_name) in &expected {
                    if !declared_set.contains(expected_name.as_str()) {
                        return Err(InferenceError::Inference(format!(
                            "OnlineArtifactDescriptor: V1Online artifact's \
                             {id:?} recipe is scoped to layer {idx}, but \
                             asymmetric_tensor_names does not declare the \
                             expected tensor {expected_name:?} — per-layer \
                             coverage is incomplete"
                        )));
                    }
                }
                let expected_set: HashSet<&str> =
                    expected.iter().map(|(_, _, e)| e.as_str()).collect();
                for declared in &self.asymmetric_tensor_names {
                    if !expected_set.contains(declared.as_str()) {
                        let expected_names: Vec<&str> =
                            expected.iter().map(|(_, _, e)| e.as_str()).collect();
                        return Err(InferenceError::Inference(format!(
                            "OnlineArtifactDescriptor: asymmetric_tensor_names \
                             declares {declared:?}, which does not match any \
                             tensor this recipe's online rotations counter-rotate \
                             ({expected_names:?}) — an unrelated declaration must \
                             not satisfy the fail-closed asymmetric-Q4 requirement"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Streaming SafeTensors reader for the QuaRot conversion pipeline.
///
/// See module documentation for layout detection and caching semantics.
#[derive(Debug)]
pub struct QuarotTensorReader {
    backing: Backing,
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
        let single_path = model_dir.join("model.safetensors");
        let index_path = model_dir.join("model.safetensors.index.json");

        if single_path.exists() {
            let shard = Shard::open(&single_path)?;
            Ok(Self {
                backing: Backing::Single { shard },
            })
        } else if index_path.exists() {
            let index = parse_index(model_dir)?;
            let mut shards: HashMap<String, Shard> = HashMap::new();
            for shard_file in index.weight_map.values() {
                if shards.contains_key(shard_file) {
                    continue;
                }
                // Index-declared shard names are untrusted checkpoint content;
                // containment-check before mapping (#1069).
                let shard = Shard::open(&contained_shard_path(model_dir, shard_file)?)?;
                shards.insert(shard_file.clone(), shard);
            }
            Ok(Self {
                backing: Backing::Sharded {
                    weight_map: index.weight_map,
                    shards,
                },
            })
        } else {
            Err(InferenceError::InvalidSafetensors(format!(
                "{}: missing both model.safetensors and \
                 model.safetensors.index.json",
                model_dir.display()
            )))
        }
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
        let data = decode_bytes_to_f64(bytes, header.dtype)?;
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

fn decode_bytes_to_f64(bytes: &[u8], dtype: SourceDType) -> Result<Vec<f64>, InferenceError> {
    match dtype {
        SourceDType::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "F32 tensor byte length {} not divisible by 4",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
                .collect())
        }
        SourceDType::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "BF16 tensor byte length {} not divisible by 2",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| {
                    crate::weights::half_bits::bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]))
                        as f64
                })
                .collect())
        }
        SourceDType::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "F16 tensor byte length {} not divisible by 2",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| {
                    crate::weights::half_bits::f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]))
                        as f64
                })
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quarot::plan::{AbsorptionSide, RotationId};
    use std::fs::File;
    use std::io::Write;

    /// On-disk dtype for synthetic test fixtures.
    #[derive(Copy, Clone)]
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
    fn fnuz_dtype_tensors_are_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        // F8_E4M3FNUZ / F8_E5M2FNUZ are official SafeTensors dtypes this
        // reader does not decode. Opening a checkpoint that merely contains
        // one must still succeed and expose the readable tensors — it must
        // not error out as an unrecognized dtype.
        let header = serde_json::json!({
            "supported": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
            "e4m3fnuz": {
                "dtype": "F8_E4M3FNUZ",
                "shape": [1],
                "data_offsets": [4, 5],
            },
            "e5m2fnuz": {
                "dtype": "F8_E5M2FNUZ",
                "shape": [1],
                "data_offsets": [5, 6],
            },
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend_from_slice(&7f32.to_le_bytes());
        buf.extend_from_slice(&[0u8, 0u8]);
        std::fs::write(dir.path().join("model.safetensors"), &buf).unwrap();

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("supported"));
        assert!(!reader.has_tensor("e4m3fnuz"));
        assert!(!reader.has_tensor("e5m2fnuz"));
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

    /// #1069: a weight_map entry that escapes the model directory (here via
    /// `..` traversal to a structurally valid shard outside it) must be
    /// rejected at open time, before anything is memory-mapped.
    #[test]
    fn sharded_index_entry_escaping_model_dir_is_rejected() {
        let outer = tempfile::tempdir().unwrap();
        let model_dir = outer.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        write_safetensors(
            &outer.path().join("evil.safetensors"),
            &[("t.weight", FixtureDType::F32, vec![1], &[1.0])],
        );
        let index = serde_json::json!({
            "metadata": {"total_size": 4usize},
            "weight_map": {"t.weight": "../evil.safetensors"},
        });
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let err = QuarotTensorReader::open(&model_dir).unwrap_err();
        match err {
            InferenceError::InvalidSafetensors(msg) => {
                assert!(msg.contains("outside the model directory"), "got: {msg}");
            }
            other => panic!("expected InvalidSafetensors, got {other:?}"),
        }
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

    // --- ArtifactVersion / OnlineArtifactDescriptor (issue #703 PR1) ---
    //
    // `ArtifactVersion`'s `#[serde(rename = ...)]` attributes are the single
    // on-disk spelling of each tag; deserializing a JSON string through the
    // derived `Deserialize` impl below is the one parser, exercised
    // directly rather than through a second hand-written match.

    #[test]
    fn artifact_version_deserializes_known_tags() {
        assert_eq!(
            serde_json::from_str::<ArtifactVersion>("\"v0-residual\"").unwrap(),
            ArtifactVersion::V0Residual
        );
        assert_eq!(
            serde_json::from_str::<ArtifactVersion>("\"v1-online-r3r4\"").unwrap(),
            ArtifactVersion::V1Online
        );
    }

    #[test]
    fn artifact_version_refuses_unknown_tag() {
        assert!(serde_json::from_str::<ArtifactVersion>("\"v2-future\"").is_err());
    }

    #[test]
    fn artifact_version_refuses_empty_tag() {
        assert!(serde_json::from_str::<ArtifactVersion>("\"\"").is_err());
    }

    #[test]
    fn artifact_version_refuses_case_variant() {
        // Refuses rather than normalizing — an unexpected casing is
        // evidence of a foreign/corrupted writer, not a legitimate variant.
        assert!(serde_json::from_str::<ArtifactVersion>("\"V0-Residual\"").is_err());
    }

    fn sample_r3_cfg() -> Qwen35Config {
        crate::model::qwen35_config::Qwen35Config::qwen35_0_8b()
    }

    fn sample_r3_spec() -> OnlineRotationSpec {
        let cfg = sample_r3_cfg();
        // block_size == num_attention_heads (8): one dense cross-head
        // Hadamard covering every head (QuaRot Eq. 9's H_num_heads).
        OnlineRotationSpec::r3_full_attention(&cfg, 42, 8).unwrap()
    }

    /// Every tensor name `sample_r3_spec()`'s scope (`[3,7,11,15,19,23]`)
    /// requires full coverage for — `model.language_model.layers.{i}.self_attn.o_proj.weight`
    /// for each scoped layer index.
    fn full_r3_layer_names() -> Vec<String> {
        [3usize, 7, 11, 15, 19, 23]
            .iter()
            .map(|i| format!("model.language_model.layers.{i}.self_attn.o_proj.weight"))
            .collect()
    }

    #[test]
    fn v0_residual_with_no_online_rotations_is_valid() {
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V0Residual,
            online_rotations: Vec::new(),
            asymmetric_tensor_names: Vec::new(),
        };
        assert!(descriptor.validate(None).is_ok());
    }

    /// A V0Residual descriptor must never carry online rotation metadata —
    /// feeding it one must be refused loudly rather than silently ignored
    /// (which would let a hand-edited index sneak online-rotation intent
    /// past a v0-only loader).
    #[test]
    fn v0_residual_with_online_rotation_is_refused() {
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V0Residual,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: Vec::new(),
        };
        let err = descriptor.validate(None).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("never interpreted as online-capable"),
            "got: {msg}"
        );
    }

    /// A V0Residual descriptor must never carry
    /// asymmetric-Q4 tensor declarations — the V0 contract is symmetric-only
    /// Q4 with no R3/R4 metadata, so a nonempty `asymmetric_tensor_names`
    /// list on a V0 descriptor is evidence of a corrupted or hand-edited
    /// index, same as a nonempty `online_rotations` list.
    #[test]
    fn v0_residual_with_asymmetric_tensor_names_is_refused() {
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V0Residual,
            online_rotations: Vec::new(),
            asymmetric_tensor_names: vec![
                "model.language_model.layers.3.self_attn.o_proj.weight".to_string(),
            ],
        };
        let err = descriptor.validate(None).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("must not declare any asymmetric_tensor_names"),
            "got: {msg}"
        );
    }

    /// Mutation/refusal: V1Online with an empty rotation list is a
    /// contradiction (an "online" artifact that rotates nothing).
    #[test]
    fn v1_online_with_zero_rotations_is_refused() {
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: Vec::new(),
            asymmetric_tensor_names: Vec::new(),
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("at least one R3/R4"), "got: {msg}");
    }

    /// `validate` must refuse immediately for a
    /// `V1Online` descriptor called without `cfg`, rather than silently
    /// running a weaker (per-layer-coverage-skipping) check. This must hold
    /// even for a descriptor that would otherwise validate cleanly with
    /// `cfg` supplied — a caller cannot opt into the weaker semantics just
    /// by omitting the config.
    #[test]
    fn v1_online_validate_without_cfg_is_refused() {
        let names = full_r3_layer_names();
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: names,
        };
        let err = descriptor.validate(None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("requires model config"), "got: {msg}");
        // The same descriptor passes once cfg is supplied — proving the
        // None case above was refused for lack of cfg, not some other
        // reason.
        assert!(descriptor.validate(Some(&sample_r3_cfg())).is_ok());
    }

    /// Mutation/refusal: a declaration missing one scoped layer's tensor
    /// must be refused — the single `asymmetric_tensor_names` declaration
    /// must equal the recipe-derived set exactly: this is the same
    /// requirement the pre-collapse
    /// `affected_tensor_names` cross-check enforced, now checked entirely
    /// from the descriptor's own fields).
    #[test]
    fn v1_online_missing_asymmetric_declaration_is_refused() {
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: vec![
                "model.language_model.layers.3.self_attn.o_proj.weight".to_string(),
            ],
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("coverage is incomplete"), "got: {msg}");
        assert!(msg.contains("layer 7"), "got: {msg}");
    }

    /// Preserved under the single-representation
    /// contract: a real rotation recipe with zero asymmetric declarations
    /// must be refused internally rather than silently accepted.
    #[test]
    fn v1_online_empty_asymmetric_list_is_refused() {
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: vec![],
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("self-contradictory"), "got: {msg}");
    }

    /// A descriptor whose caller-supplied vectors exceed the validation cap is
    /// rejected up front, before any per-entry scan, so validation cost stays
    /// bounded regardless of how large an untrusted descriptor is.
    #[test]
    fn validate_rejects_descriptor_exceeding_entry_cap() {
        let over_cap = OnlineArtifactDescriptor::MAX_VALIDATED_ENTRIES + 1;
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: (0..over_cap).map(|i| format!("t{i}")).collect(),
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("exceeding the maximum"), "got: {msg}");
    }

    /// Regression test, preserved under the
    /// single-representation contract: a real R3 recipe declaring an
    /// unrelated, nonempty `asymmetric_tensor_names` entry must be rejected
    /// for not matching any tensor the recipe's derived expected set
    /// contains — an unrelated string must not satisfy the fail-closed
    /// requirement just by being nonempty.
    #[test]
    fn v1_online_unrelated_asymmetric_declaration_is_refused() {
        // Full, correct per-layer coverage PLUS one unrelated extra entry —
        // isolates the extra-direction (unrelated) check from the
        // missing-direction check exercised by
        // `v1_online_missing_asymmetric_declaration_is_refused` above.
        let mut names = full_r3_layer_names();
        names.push("unrelated".to_string());
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: names,
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unrelated") && msg.contains("does not match any tensor"),
            "got: {msg}"
        );
    }

    /// A declared name with the right suffix but an out-of-scope layer
    /// (layer 0 is GDN, outside `sample_r3_spec()`'s full-attention scope)
    /// must also be rejected as "extra" — exact-set equality, not a
    /// suffix-only membership check.
    #[test]
    fn v1_online_out_of_scope_layer_declaration_is_refused() {
        let mut names = full_r3_layer_names();
        names.push("model.language_model.layers.0.self_attn.o_proj.weight".to_string());
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: names,
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("layers.0.self_attn.o_proj.weight")
                && msg.contains("does not match any tensor"),
            "got: {msg}"
        );
    }

    #[test]
    fn v1_online_with_full_asymmetric_coverage_is_valid() {
        // Regression test: a complete, exact declaration
        // (every scoped layer's tensor, nothing extra) must pass.
        let names = full_r3_layer_names();
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: names,
        };
        // cfg is mandatory for V1Online — see
        // `v1_online_validate_without_cfg_is_refused` for the None case.
        assert!(descriptor.validate(Some(&sample_r3_cfg())).is_ok());
    }

    // --- OnlineRotationSpec::validate ---

    #[test]
    fn online_rotation_spec_accepts_constructor_built_specs() {
        let cfg = sample_r3_cfg();
        assert!(sample_r3_spec().validate(None).is_ok());
        assert!(sample_r3_spec().validate(Some(&cfg)).is_ok());
        let r4 = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 256).unwrap();
        assert!(r4.validate(None).is_ok());
        assert!(r4.validate(Some(&cfg)).is_ok());
    }

    /// The exact malformed spec: an R3 id
    /// with `OutputSide`, `block_size=3` (not a power of two), and
    /// `layer_scope=None` (R3 requires Some). Directly constructed —
    /// bypassing `r3_full_attention` — because that is precisely how a
    /// hand-edited or corrupted index could produce this. Must be rejected
    /// both standalone and via the descriptor path.
    #[test]
    fn online_rotation_spec_rejects_the_reported_malformed_r3_spec() {
        let malformed = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::OutputSide,
            seed: 1,
            block_size: 3,
            layer_scope: None,
        };
        assert!(malformed.validate(None).is_err());

        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![malformed],
            asymmetric_tensor_names: vec![
                "model.language_model.layers.3.self_attn.o_proj.weight".to_string(),
            ],
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        assert!(
            format!("{err}").contains("InputSide"),
            "descriptor validation must surface the spec-level rejection, got: {err}"
        );
    }

    #[test]
    fn online_rotation_spec_rejects_wrong_side() {
        let spec = OnlineRotationSpec {
            id: RotationId::MlpDownR4,
            side: AbsorptionSide::OutputSide,
            seed: 1,
            block_size: 64,
            layer_scope: None,
        };
        let err = spec.validate(None).unwrap_err();
        assert!(format!("{err}").contains("InputSide"));
    }

    #[test]
    fn online_rotation_spec_rejects_non_power_of_two_block_size() {
        let spec = OnlineRotationSpec {
            id: RotationId::MlpDownR4,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 96,
            layer_scope: None,
        };
        let err = spec.validate(None).unwrap_err();
        assert!(format!("{err}").contains("power-of-two"));
    }

    #[test]
    fn online_rotation_spec_rejects_zero_block_size() {
        let spec = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 0,
            layer_scope: Some(vec![3]),
        };
        let err = spec.validate(None).unwrap_err();
        assert!(format!("{err}").contains("power-of-two"));
    }

    #[test]
    fn online_rotation_spec_rejects_r3_with_none_layer_scope() {
        let spec = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: None,
        };
        let err = spec.validate(None).unwrap_err();
        assert!(format!("{err}").contains("layer_scope"));
    }

    #[test]
    fn online_rotation_spec_rejects_r3_with_empty_layer_scope() {
        let spec = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: Some(vec![]),
        };
        let err = spec.validate(None).unwrap_err();
        assert!(format!("{err}").contains("must not be empty"));
    }

    #[test]
    fn online_rotation_spec_rejects_r4_with_some_layer_scope() {
        let spec = OnlineRotationSpec {
            id: RotationId::MlpDownR4,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 64,
            layer_scope: Some(vec![0]),
        };
        let err = spec.validate(None).unwrap_err();
        assert!(format!("{err}").contains("must be None"));
    }

    #[test]
    fn online_rotation_spec_rejects_residual_stream_id() {
        let spec = OnlineRotationSpec {
            id: RotationId::ResidualStream,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: None,
        };
        let err = spec.validate(None).unwrap_err();
        assert!(format!("{err}").contains("ResidualStream"));
    }

    #[test]
    fn online_rotation_spec_cfg_aware_checks_catch_bad_divisibility_and_scope() {
        let cfg = sample_r3_cfg();
        // block_size divides nothing meaningful without cfg, but with cfg
        // it must divide num_attention_heads (8) — 4 does. layer_scope must
        // be the config's full-attention layers exactly, so this uses the
        // full set (not a subset — see the equality test below).
        let good_block = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 4, // power of two, divides 8 — valid without extra checks
            layer_scope: Some(vec![3, 7, 11, 15, 19, 23]),
        };
        assert!(good_block.validate(Some(&cfg)).is_ok());

        // A layer_scope entry that is not actually a full-attention layer
        // (layer 0 is GDN in qwen35_0_8b) must be rejected once cfg is known.
        let bad_scope = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: Some(vec![0]),
        };
        assert!(!cfg.is_full_attention(0), "sanity: layer 0 is GDN");
        let err = bad_scope.validate(Some(&cfg)).unwrap_err();
        assert!(format!("{err}").contains("layer 0"));
    }

    /// The cfg-aware check must reject a
    /// `layer_scope` that is a strict subset of the config's full-attention
    /// layers — membership alone is not sufficient, coverage must be
    /// complete. For `qwen35_0_8b`, full-attention layers are
    /// `[3,7,11,15,19,23]`; a scope of just `[3]` must be rejected, naming
    /// the missing layers.
    #[test]
    fn online_rotation_spec_rejects_r3_scope_missing_full_attention_layers() {
        let cfg = sample_r3_cfg();
        let partial_scope = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: Some(vec![3]),
        };
        let err = partial_scope.validate(Some(&cfg)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("missing"), "got: {msg}");
        for missing in ["7", "11", "15", "19", "23"] {
            assert!(msg.contains(missing), "expected {missing} in: {msg}");
        }

        // The full set passes.
        let full_scope = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: Some(vec![3, 7, 11, 15, 19, 23]),
        };
        assert!(full_scope.validate(Some(&cfg)).is_ok());

        // A superset naming a non-full-attention layer is still rejected
        // (the membership direction, unaffected by the equality fix).
        let superset_with_non_member = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: Some(vec![0, 3, 7, 11, 15, 19, 23]),
        };
        let err = superset_with_non_member.validate(Some(&cfg)).unwrap_err();
        assert!(format!("{err}").contains("layer 0"));
    }

    /// A scope with a duplicate layer index (the
    /// exact `[3,3,7,11,15,19,23]` example from the review) must be
    /// rejected outright — a naive set-equality check would silently dedup
    /// it and accept. The check is cfg-independent, so it must also fire
    /// with `cfg: None`.
    #[test]
    fn online_rotation_spec_rejects_duplicate_r3_layer_index() {
        let duplicate_scope = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: Some(vec![3, 3, 7, 11, 15, 19, 23]),
        };
        let err = duplicate_scope.validate(None).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("strictly sorted ascending") && msg.contains("duplicates"),
            "got: {msg}"
        );
        let err = duplicate_scope
            .validate(Some(&sample_r3_cfg()))
            .unwrap_err();
        assert!(
            format!("{err}").contains("strictly sorted ascending"),
            "duplicate rejection must fire before/independent of cfg checks"
        );
    }

    /// An unsorted (but duplicate-free) scope must
    /// also be rejected — the canonical form is strictly ascending, not
    /// merely a valid set.
    #[test]
    fn online_rotation_spec_rejects_unsorted_r3_layer_scope() {
        let unsorted_scope = OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 1,
            block_size: 8,
            layer_scope: Some(vec![7, 3, 11, 15, 19, 23]),
        };
        let err = unsorted_scope.validate(None).unwrap_err();
        assert!(
            format!("{err}").contains("strictly sorted ascending"),
            "got: {err}"
        );
    }

    /// A V1Online descriptor must not accept two
    /// online rotation specs targeting the same runtime site — two
    /// `AttentionOutputR3` entries (even with different seeds) leave the
    /// future runtime to invent duplicate-site semantics, so the second one
    /// is rejected.
    #[test]
    fn v1_online_rejects_two_specs_targeting_the_same_site() {
        let cfg = sample_r3_cfg();
        let first = OnlineRotationSpec::r3_full_attention(&cfg, 42, 8).unwrap();
        let second = OnlineRotationSpec::r3_full_attention(&cfg, 99, 8).unwrap();
        let names = full_r3_layer_names();
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![first, second],
            asymmetric_tensor_names: names,
        };
        let err = descriptor.validate(Some(&cfg)).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("same runtime site") && msg.contains("AttentionOutputPreOProj"),
            "got: {msg}"
        );
    }

    /// A well-formed single-spec-per-site descriptor (R3 + R4 together)
    /// must still pass — the same-site rejection must not false-positive
    /// on two specs that target *different* sites.
    #[test]
    fn v1_online_with_distinct_sites_is_valid() {
        let cfg = sample_r3_cfg();
        let r3 = OnlineRotationSpec::r3_full_attention(&cfg, 42, 8).unwrap();
        let r4 = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 256).unwrap();
        let mut names = full_r3_layer_names();
        for i in 0..cfg.num_hidden_layers {
            names.push(format!(
                "{}.mlp.down_proj.weight",
                crate::model::qwen35::qwen_layer_tensor_prefix(i)
            ));
        }
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![r3, r4],
            asymmetric_tensor_names: names,
        };
        assert!(descriptor.validate(Some(&cfg)).is_ok());
    }

    // --- canonical Qwen3.5 tensor namespace ---

    /// The canonical Qwen3.5 tensor namespace is
    /// `model.language_model.layers.{idx}.{suffix}` (matching
    /// `qwen_required_tensor_names` and the QuaRot converter), not
    /// `model.layers.{idx}.{suffix}`. A descriptor declaring the canonical
    /// names for the full R3 scope must pass; the same descriptor rewritten
    /// with the old, non-canonical prefix must now fail per-layer coverage.
    #[test]
    fn v1_online_per_layer_coverage_uses_canonical_qwen_namespace() {
        let cfg = sample_r3_cfg();
        let canonical_names = full_r3_layer_names();
        for name in &canonical_names {
            assert!(
                name.starts_with("model.language_model.layers."),
                "got: {name}"
            );
        }
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: canonical_names,
        };
        assert!(descriptor.validate(Some(&cfg)).is_ok());

        let old_namespace_names: Vec<String> = [3usize, 7, 11, 15, 19, 23]
            .iter()
            .map(|i| format!("model.layers.{i}.self_attn.o_proj.weight"))
            .collect();
        let old_descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: old_namespace_names,
        };
        let err = old_descriptor.validate(Some(&cfg)).unwrap_err();
        assert!(
            format!("{err}").contains("coverage is incomplete"),
            "old model.layers.* namespace must be rejected as incomplete coverage"
        );
    }

    // --- per-layer tensor coverage ---

    /// The exact scenario from the finding: an R3 recipe scoped to layers
    /// `[3,7,11,15,19,23]` must not validate against a declaration that only
    /// contains layer 0's tensor (suffix matches, layer doesn't). Must fail
    /// naming a missing scoped layer.
    #[test]
    fn v1_online_layer_scope_coverage_rejects_wrong_layer_declaration() {
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: vec![
                "model.language_model.layers.0.self_attn.o_proj.weight".to_string(),
            ],
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("layer 3") && msg.contains("coverage is incomplete"),
            "got: {msg}"
        );
    }

    /// Partial coverage — every scoped layer but one declared — must still
    /// be rejected, naming the missing layer.
    #[test]
    fn v1_online_layer_scope_coverage_rejects_partial_declaration() {
        let mut names = full_r3_layer_names();
        names.pop(); // drop layer 23
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: names,
        };
        let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("layer 23") && msg.contains("coverage is incomplete"),
            "got: {msg}"
        );
    }

    /// R4 (`layer_scope: None`, "every layer"
    /// semantics) validation without `cfg` is refused outright — the prior
    /// behavior of silently *skipping* the per-layer coverage check for a
    /// `None`-scoped rotation (because the full layer count was
    /// "unknowable") was the exact bug the review escalated: a V1/R4
    /// descriptor declaring only `model.language_model.layers.0.mlp.down_proj.weight` must
    /// no longer pass any validation path that claims completeness, with or
    /// without cfg games. With `cfg` supplied, the same partial declaration
    /// is rejected for missing coverage of the layers `cfg` says exist.
    #[test]
    fn v1_online_r4_partial_declaration_rejected_regardless_of_cfg() {
        let cfg = sample_r3_cfg();
        let r4 = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 256).unwrap();
        let partial_name = "model.language_model.layers.0.mlp.down_proj.weight".to_string();
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![r4],
            asymmetric_tensor_names: vec![partial_name],
        };
        // Without cfg: refused immediately — no path claims completeness
        // without model config.
        let err = descriptor.validate(None).unwrap_err();
        assert!(format!("{err}").contains("requires model config"));
        // With cfg: full-layer coverage is enforced and layer 1..N are
        // missing, so validation must fail here too.
        let err = descriptor.validate(Some(&cfg)).unwrap_err();
        assert!(format!("{err}").contains("coverage is incomplete"));
    }

    // --- verify_affected (narrow converter cross-check entry point) ---

    #[test]
    fn verify_affected_accepts_exact_match() {
        let cfg = sample_r3_cfg();
        let names = full_r3_layer_names();
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: names.clone(),
        };
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        assert!(descriptor.verify_affected(&cfg, &refs).is_ok());
    }

    #[test]
    fn verify_affected_rejects_missing_and_extra() {
        let cfg = sample_r3_cfg();
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![sample_r3_spec()],
            asymmetric_tensor_names: full_r3_layer_names(),
        };
        // Missing: converter reports transforming nothing.
        assert!(descriptor.verify_affected(&cfg, &[]).is_err());
        // Extra: converter reports an unrelated tensor alongside full coverage.
        let mut names = full_r3_layer_names();
        names.push("unrelated".to_string());
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        assert!(descriptor.verify_affected(&cfg, &refs).is_err());
    }
}
