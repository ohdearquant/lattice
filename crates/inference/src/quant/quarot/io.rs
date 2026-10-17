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
//! conversion in `crate::weights::half_bits` (independent of the `f16`
//! cargo feature, which only gates *loading permission* elsewhere), then
//! widens the result to f64 (lossless from f32).
//!
//! Step 3b deliverable per [ADR-044]; consumed by step 3c's
//! `quantize_quarot` binary.
//!
//! [ADR-044]: ../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md

use std::collections::{HashMap, HashSet};
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

use crate::error::InferenceError;
use crate::model::qwen35::qwen_layer_tensor_prefix;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::plan::{OnlineRotationSpec, OnlineTransformSite};
use crate::weights::ingress::DecodedTensorValidator;
use crate::weights::safetensors_layout::{
    SafetensorsLayoutEntry, safetensors_dtype, validate_safetensors_layout,
};
use crate::weights::{contained_shard_path, parse_index};

/// On-disk storage dtype of a tensor.
///
/// Returned by [`QuarotTensorReader::source_dtype`] so the converter can
/// record provenance in its output metadata. The converter always promotes
/// to f64 internally; this is informational, not a control knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    source: String,
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
    dtype: &'static str,
    shape: Vec<usize>,
    start: usize,
    end: usize,
    /// `Some` for F32/F16/BF16; `None` for any other SafeTensors dtype.
    /// `None` entries are still kept in the contiguity check but never
    /// appear in `Shard::headers`.
    header: Option<TensorHeader>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, InferenceError> {
        let file = crate::weights::mmap_trust::open_regular_file_no_hang(path)
            .map_err(InferenceError::InvalidSafetensors)?;
        // This reader loads the same HuggingFace hub-cache checkpoints the
        // runtime baseline forward pass does (see `QuarotTensorReader::open`'s
        // doc comment), so `path` may resolve through a final-component
        // symlink -- `open_trusted_mmap_file`'s `O_NOFOLLOW` would reject
        // that. `map_after_untrusted_open` applies the same
        // mode/uid/ACL/parent-directory checks without it, then maps.
        let mmap = crate::weights::mmap_trust::map_after_untrusted_open(&file, path)
            .map_err(InferenceError::InvalidSafetensors)?;

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
        // doesn't decode — so the shared validator can account for the
        // complete data section.
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

            let dtype = safetensors_dtype(dtype_str).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: unrecognized SafeTensors dtype {dtype_str:?}"
                ))
            })?;

            let header = SourceDType::from_header_str(dtype_str).map(|dtype| TensorHeader {
                dtype,
                shape: shape.clone(),
                start,
                end,
            });

            parsed.push(ParsedEntry {
                name: name.clone(),
                dtype: dtype.name,
                shape,
                start,
                end,
                header,
            });
        }

        // Phase 2: apply the same dtype extent, byte-alignment, and complete
        // data-section contract as the runtime SafeTensors loader.
        let layout: Vec<_> = parsed
            .iter()
            .map(|entry| SafetensorsLayoutEntry {
                name: &entry.name,
                dtype: entry.dtype,
                shape: &entry.shape,
                start: entry.start,
                end: entry.end,
            })
            .collect();
        validate_safetensors_layout(&path.display().to_string(), data_len, &layout)?;

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
            source: path.display().to_string(),
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
#[non_exhaustive]
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
        let data = decode_bytes_to_f64(bytes, header.dtype, &shard.source, name, &header.shape)?;
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

const VALIDATED_DECODE_LANES: usize = 16;

fn decode_bytes_to_f64(
    bytes: &[u8],
    dtype: SourceDType,
    source: &str,
    tensor_name: &str,
    shape: &[usize],
) -> Result<Vec<f64>, InferenceError> {
    let element_width = match dtype {
        SourceDType::F32 => 4,
        SourceDType::F16 | SourceDType::BF16 => 2,
    };
    if !bytes.len().is_multiple_of(element_width) {
        return Err(InferenceError::InvalidSafetensors(format!(
            "{source}: tensor {tensor_name} ({}) byte length {} not divisible by \
             element width {element_width}",
            dtype.name(),
            bytes.len(),
        )));
    }

    let decoded_len = bytes.len() / element_width;
    let validator =
        DecodedTensorValidator::safetensors(source, tensor_name, shape, dtype.name(), decoded_len)?;
    let mut data = Vec::with_capacity(decoded_len);
    let mut non_finite_lanes = [false; VALIDATED_DECODE_LANES];

    match dtype {
        SourceDType::F32 => {
            let mut groups = bytes.chunks_exact(VALIDATED_DECODE_LANES * 4);
            for group in &mut groups {
                let mut decoded = [0.0_f64; VALIDATED_DECODE_LANES];
                macro_rules! decode_lane {
                    ($lane:literal) => {{
                        let offset = $lane * 4;
                        let bits = u32::from_le_bytes([
                            group[offset],
                            group[offset + 1],
                            group[offset + 2],
                            group[offset + 3],
                        ]);
                        non_finite_lanes[$lane] |= bits & 0x7f80_0000 == 0x7f80_0000;
                        decoded[$lane] = f32::from_bits(bits) as f64;
                    }};
                }
                decode_lane!(0);
                decode_lane!(1);
                decode_lane!(2);
                decode_lane!(3);
                decode_lane!(4);
                decode_lane!(5);
                decode_lane!(6);
                decode_lane!(7);
                decode_lane!(8);
                decode_lane!(9);
                decode_lane!(10);
                decode_lane!(11);
                decode_lane!(12);
                decode_lane!(13);
                decode_lane!(14);
                decode_lane!(15);
                data.extend_from_slice(&decoded);
            }
            for bytes in groups.remainder().chunks_exact(4) {
                let bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                non_finite_lanes[0] |= bits & 0x7f80_0000 == 0x7f80_0000;
                data.push(f32::from_bits(bits) as f64);
            }
        }
        SourceDType::BF16 => {
            let mut groups = bytes.chunks_exact(VALIDATED_DECODE_LANES * 2);
            for group in &mut groups {
                let mut decoded = [0.0_f64; VALIDATED_DECODE_LANES];
                macro_rules! decode_lane {
                    ($lane:literal) => {{
                        let offset = $lane * 2;
                        let bits = u16::from_le_bytes([group[offset], group[offset + 1]]);
                        non_finite_lanes[$lane] |= bits & 0x7f80 == 0x7f80;
                        decoded[$lane] = crate::weights::half_bits::bf16_bits_to_f32(bits) as f64;
                    }};
                }
                decode_lane!(0);
                decode_lane!(1);
                decode_lane!(2);
                decode_lane!(3);
                decode_lane!(4);
                decode_lane!(5);
                decode_lane!(6);
                decode_lane!(7);
                decode_lane!(8);
                decode_lane!(9);
                decode_lane!(10);
                decode_lane!(11);
                decode_lane!(12);
                decode_lane!(13);
                decode_lane!(14);
                decode_lane!(15);
                data.extend_from_slice(&decoded);
            }
            for bytes in groups.remainder().chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                non_finite_lanes[0] |= bits & 0x7f80 == 0x7f80;
                data.push(crate::weights::half_bits::bf16_bits_to_f32(bits) as f64);
            }
        }
        SourceDType::F16 => {
            let mut groups = bytes.chunks_exact(VALIDATED_DECODE_LANES * 2);
            for group in &mut groups {
                let mut decoded = [0.0_f64; VALIDATED_DECODE_LANES];
                macro_rules! decode_lane {
                    ($lane:literal) => {{
                        let offset = $lane * 2;
                        let bits = u16::from_le_bytes([group[offset], group[offset + 1]]);
                        non_finite_lanes[$lane] |= bits & 0x7c00 == 0x7c00;
                        decoded[$lane] = crate::weights::half_bits::f16_bits_to_f32(bits) as f64;
                    }};
                }
                decode_lane!(0);
                decode_lane!(1);
                decode_lane!(2);
                decode_lane!(3);
                decode_lane!(4);
                decode_lane!(5);
                decode_lane!(6);
                decode_lane!(7);
                decode_lane!(8);
                decode_lane!(9);
                decode_lane!(10);
                decode_lane!(11);
                decode_lane!(12);
                decode_lane!(13);
                decode_lane!(14);
                decode_lane!(15);
                data.extend_from_slice(&decoded);
            }
            for bytes in groups.remainder().chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                non_finite_lanes[0] |= bits & 0x7c00 == 0x7c00;
                data.push(crate::weights::half_bits::f16_bits_to_f32(bits) as f64);
            }
        }
    }

    let has_non_finite = non_finite_lanes.into_iter().any(|found| found);
    validator.finish_f64_materialization(&data, has_non_finite)?;
    Ok(data)
}

#[cfg(test)]
mod tests;
