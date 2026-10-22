//! Safetensors metadata and f32 weight structs, including tensor, transformer-layer, BERT, cross-encoder, safetensors-file, and Qwen layer weights.
use crate::bounded_read::{BoundedReadError, read_text_bounded};
use crate::error::InferenceError;
use crate::weights::safetensors_layout::{
    SafetensorsLayoutEntry, safetensors_dtype, validate_safetensors_layout,
};
use memmap2::Mmap;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
    F16,
    BF16,
    /// OCP FP8 E4M3 ("FN": finite-only, no infinity), safetensors `F8_E4M3`.
    /// Distinct from `F8_E4M3FNUZ`, which stays `Other` (lattice#684).
    F8E4M3,
    /// OCP FP8 E5M2, safetensors `F8_E5M2`. Distinct from `F8_E5M2FNUZ`,
    /// which stays `Other` (lattice#684).
    F8E5M2,
    /// A dtype safetensors defines but this crate does not materialize as
    /// f32 (I64, U8, BOOL, F64, ...). Tracked structurally (not silently
    /// dropped from `SafetensorsFile::tensors`) so extent validation still
    /// runs at open time; only a *requested* tensor of this dtype fails,
    /// via `get_f32_tensor` (lattice#800).
    Other {
        label: &'static str,
    },
}

impl DType {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::Other { label, .. } => label,
        }
    }
}

/// Resolve a safetensors header `dtype` string to a [`DType`], including
/// non-float dtypes (tracked via `DType::Other` so extent validation still
/// covers them). Returns `None` only for a string this crate has never
/// heard of — the caller treats that as a hard parse error, since an
/// unrecognized dtype's byte width is unknowable and extent validation
/// cannot proceed (lattice#800).
fn dtype_from_str(s: &str) -> Option<DType> {
    Some(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F8_E4M3" => DType::F8E4M3,
        "F8_E5M2" => DType::F8E5M2,
        _ => {
            let dtype = safetensors_dtype(s)?;
            DType::Other { label: dtype.name }
        }
    })
}

#[derive(Debug)]
struct TensorMeta {
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
    converted_f32: OnceLock<Box<[f32]>>,
    /// Populated once, via `get_or_init`, by this tensor's ingress-validation
    /// outcome. Zero-copy F32 uses the decoded-slice validator; F16/BF16
    /// widening publishes its reduction here before the cached slice escapes.
    /// Concurrent first access therefore cannot validate the same tensor
    /// twice. Ingress validation only returns
    /// `InferenceError::InvalidSafetensors` on this path, so the cached error
    /// is carried as its message and rewrapped on read.
    validated: OnceLock<Result<(), String>>,
}

/// **Unstable**: 2D tensor view over memory-mapped safetensors data; field layout may change.
///
/// A 2D tensor (matrix) backed by memory-mapped data.
#[derive(Debug, Clone, Copy)]
pub struct Tensor2D<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

/// **Unstable**: 1D tensor view over memory-mapped safetensors data; field layout may change.
///
/// A 1D tensor (vector/bias) backed by memory-mapped data.
#[derive(Debug, Clone, Copy)]
pub struct Tensor1D<'a> {
    pub data: &'a [f32],
    pub len: usize,
}

/// **Unstable**: per-layer BERT transformer weights; field set may change with model variants.
///
/// All weights for one transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayerWeights<'a> {
    pub query_weight: Tensor2D<'a>,
    pub query_bias: Tensor1D<'a>,
    pub key_weight: Tensor2D<'a>,
    pub key_bias: Tensor1D<'a>,
    pub value_weight: Tensor2D<'a>,
    pub value_bias: Tensor1D<'a>,
    pub attn_output_weight: Tensor2D<'a>,
    pub attn_output_bias: Tensor1D<'a>,
    pub attn_layer_norm_weight: Tensor1D<'a>,
    pub attn_layer_norm_bias: Tensor1D<'a>,
    pub ffn_intermediate_weight: Tensor2D<'a>,
    pub ffn_intermediate_bias: Tensor1D<'a>,
    pub ffn_output_weight: Tensor2D<'a>,
    pub ffn_output_bias: Tensor1D<'a>,
    pub ffn_layer_norm_weight: Tensor1D<'a>,
    pub ffn_layer_norm_bias: Tensor1D<'a>,
}

/// **Unstable**: full BERT model weight bundle; structure may change with model variants.
///
/// All weights for the BERT model.
#[derive(Debug, Clone)]
pub struct BertWeights<'a> {
    pub word_embeddings: Tensor2D<'a>,
    pub position_embeddings: Tensor2D<'a>,
    pub token_type_embeddings: Tensor2D<'a>,
    pub embedding_layer_norm_weight: Tensor1D<'a>,
    pub embedding_layer_norm_bias: Tensor1D<'a>,
    pub layers: Vec<TransformerLayerWeights<'a>>,
    pub pooler_weight: Tensor2D<'a>,
    pub pooler_bias: Tensor1D<'a>,
}

/// Owned classifier head weights for a `BertForSequenceClassification` checkpoint.
///
/// Copied at load time so the struct is `'static` and independent of the mmap.
#[derive(Debug, Clone)]
pub struct CrossEncoderWeights {
    /// Classifier weight row; length equals `hidden_size`.
    pub classifier_weight: Vec<f32>,
    /// Classifier bias scalar.
    pub classifier_bias: f32,
}

impl CrossEncoderWeights {
    /// Compute the raw logit: `weight · pooled + bias`.
    pub fn logit(&self, pooled: &[f32]) -> f32 {
        debug_assert_eq!(self.classifier_weight.len(), pooled.len());
        self.classifier_weight
            .iter()
            .zip(pooled.iter())
            .map(|(w, v)| w * v)
            .sum::<f32>()
            + self.classifier_bias
    }
}

/// Backing storage for a parsed safetensors file: either a memory-mapped file
/// (the `open()` path) or an owned byte buffer (the `from_bytes()` path).
///
/// The owned variant exists for targets without real file-descriptor/mmap
/// support (`wasm32-unknown-unknown`: `memmap2`'s wasm fallback compiles but
/// every mmap construction call returns `io::ErrorKind::Unsupported` at
/// runtime) and for hosts that receive model weights as an in-memory buffer
/// rather than a filesystem path (e.g. bytes handed in from JavaScript).
enum SafetensorsBacking {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl SafetensorsBacking {
    fn as_slice(&self) -> &[u8] {
        match self {
            SafetensorsBacking::Mapped(m) => &m[..],
            SafetensorsBacking::Owned(v) => v.as_slice(),
        }
    }
}

impl std::fmt::Debug for SafetensorsBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetensorsBacking::Mapped(m) => f.debug_tuple("Mapped").field(m).finish(),
            // Vec<u8>'s derived Debug would dump every byte; report the length instead.
            SafetensorsBacking::Owned(v) => f.debug_struct("Owned").field("len", &v.len()).finish(),
        }
    }
}

/// **Unstable**: memory-mapped (or in-memory) safetensors file reader; internal representation may change.
///
/// Parsed safetensors file, backed by either a memory map or an owned byte buffer.
#[derive(Debug)]
pub struct SafetensorsFile {
    data: SafetensorsBacking,
    /// Byte offset where tensor data starts (8 + header_length).
    data_offset: usize,
    /// Parsed tensor metadata.
    tensors: HashMap<String, TensorMeta>,
    /// Human-readable provenance for this file, used only in error messages
    /// raised by the ingress validation seam (lattice#800) — the source
    /// path for `open()`, or a fixed label for `from_bytes()`.
    source: String,
}

impl SafetensorsFile {
    /// **Unstable**: open a safetensors file; error type and header parsing may change.
    ///
    /// Open and parse a safetensors file.
    pub fn open(path: &Path) -> Result<Self, InferenceError> {
        let file = crate::weights::mmap_trust::open_regular_file_no_hang(path)
            .map_err(InferenceError::InvalidSafetensors)?;
        Self::from_open_file(file, path)
    }

    /// Parse a safetensors file from an already-open [`File`].
    ///
    /// Manifest-derived shards reach this through [`open_manifest_entry_once`], which
    /// opens the file exactly once. Mapping the fd the caller already holds is what keeps
    /// that single open meaningful; reopening by path here would reintroduce the window
    /// between the open and the read. `path` is used both to run the mmap trust-boundary
    /// guard (via [`crate::weights::mmap_trust::map_after_untrusted_open`]) and to name the
    /// file in error messages -- it should be the resolved real path when the caller has
    /// one, matching that guard's own `path` requirement.
    pub(crate) fn from_open_file(file: File, path: &Path) -> Result<Self, InferenceError> {
        let display_path = path.display().to_string();
        let mmap = crate::weights::mmap_trust::map_after_untrusted_open(&file, path)
            .map_err(InferenceError::InvalidSafetensors)?;

        Self::from_backing(SafetensorsBacking::Mapped(mmap), display_path)
    }

    /// **Unstable**: parse a safetensors file already resident in memory.
    ///
    /// Unlike `open()`, this never touches `std::fs` or `mmap`, so it works on
    /// targets without real file-descriptor support (`wasm32-unknown-unknown`)
    /// and for callers that already have the model bytes in hand (e.g. bytes
    /// handed in from JavaScript across a wasm boundary).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, InferenceError> {
        Self::from_backing(
            SafetensorsBacking::Owned(bytes),
            "<in-memory bytes>".to_string(),
        )
    }

    fn from_backing(data: SafetensorsBacking, source: String) -> Result<Self, InferenceError> {
        let bytes = data.as_slice();

        if bytes.len() < 8 {
            return Err(InferenceError::InvalidSafetensors(
                "file too small to contain safetensors header".into(),
            ));
        }

        let header_len = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|_| InferenceError::InvalidSafetensors("invalid header length".into()))?,
        ) as usize;
        if header_len > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(InferenceError::InvalidSafetensors(format!(
                "header length {header_len} exceeds limit of {MAX_SAFETENSORS_HEADER_BYTES} bytes"
            )));
        }

        let data_offset = 8usize
            .checked_add(header_len)
            .ok_or_else(|| InferenceError::InvalidSafetensors("header length overflow".into()))?;

        if data_offset > bytes.len() {
            return Err(InferenceError::InvalidSafetensors(format!(
                "header extends past end of file: header_end={}, file_len={}",
                data_offset,
                bytes.len()
            )));
        }

        let header = std::str::from_utf8(&bytes[8..data_offset]).map_err(|e| {
            InferenceError::InvalidSafetensors(format!("header is not valid UTF-8: {e}"))
        })?;
        let tensors = parse_safetensors_header(header)?;

        let data_len = bytes.len() - data_offset;
        let layout: Vec<_> = tensors
            .iter()
            .map(|(name, meta)| SafetensorsLayoutEntry {
                name,
                dtype: meta.dtype.name(),
                shape: &meta.shape,
                start: meta.start,
                end: meta.end,
            })
            .collect();
        validate_safetensors_layout(&source, data_len, &layout)?;

        Ok(Self {
            data,
            data_offset,
            tensors,
            source,
        })
    }

    /// **Unstable**: query tensor shape by name; return type may change.
    ///
    /// Return a tensor shape if present.
    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|m| m.shape.as_slice())
    }

    /// **Unstable**: list all tensor names in the file; return type may change.
    ///
    /// Return a list of tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// **Unstable**: check whether a named tensor exists; method may be merged into tensor_shape.
    ///
    /// Whether the file contains a tensor.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// **Unstable**: header-declared dtype name (`"F32"`, `"F16"`, `"BF16"`,
    /// or another safetensors dtype label) for a named tensor, `None` if
    /// absent. Lets a loader enforce a checkpoint's declared dtype contract
    /// (e.g. Gemma 4's BF16-only language-model tensors) at runtime, not
    /// just in an offline header-manifest preflight.
    pub fn tensor_dtype(&self, name: &str) -> Option<&'static str> {
        self.tensors.get(name).map(|m| m.dtype.name())
    }

    fn payload_bytes(&self, meta: &TensorMeta) -> Result<&[u8], InferenceError> {
        let start = self
            .data_offset
            .checked_add(meta.start)
            .ok_or_else(|| InferenceError::InvalidSafetensors("tensor start overflow".into()))?;
        let end = self
            .data_offset
            .checked_add(meta.end)
            .ok_or_else(|| InferenceError::InvalidSafetensors("tensor end overflow".into()))?;
        Ok(&self.data.as_slice()[start..end])
    }

    /// Raw little-endian payload of a BF16 tensor, read in place from the backing storage.
    ///
    /// Nothing is widened or copied, so a table too large to hold as f32 can stay in the
    /// mapping. A non-finite value is refused with the same error `get_f32_tensor` raises for
    /// the tensor, and the result is cached like that path's.
    pub(crate) fn bf16_payload(&self, name: &str) -> Result<&[u8], InferenceError> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;
        if meta.dtype != DType::BF16 {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{}: tensor {name} has dtype {}, expected BF16",
                self.source,
                meta.dtype.name()
            )));
        }
        let bytes = self.payload_bytes(meta)?;
        let validation = meta.validated.get_or_init(|| {
            validate_bf16_payload(self.source.as_str(), name, meta.shape.as_slice(), bytes)
                .map_err(|e| e.to_string())
        });
        match validation {
            Ok(()) => Ok(bytes),
            Err(msg) => Err(InferenceError::InvalidSafetensors(msg.clone())),
        }
    }

    /// Number of tensors currently holding a cached f32 conversion.
    #[cfg(test)]
    pub(crate) fn cached_f32_tensor_count(&self) -> usize {
        self.tensors
            .values()
            .filter(|meta| meta.converted_f32.get().is_some())
            .count()
    }

    /// **Unstable**: load a named tensor as f32 slice; dtype conversion strategy may change.
    ///
    /// Get a tensor by name as f32 slice.
    pub fn get_f32_tensor(&self, name: &str) -> Result<(&[f32], &[usize]), InferenceError> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;

        let bytes = self.payload_bytes(meta)?;
        let source = self.source.as_str();
        let shape = meta.shape.as_slice();
        let dtype_name = meta.dtype.name();

        let (slice, validation_is_fused): (&[f32], bool) = match meta.dtype {
            DType::F32 => {
                // SAFETY: `open()` validated that bytes.len() is exactly the declared
                // F32 element count times 4. On little-endian targets, aligned mmap
                // storage can be viewed directly; otherwise use the owned LE conversion
                // cache below.
                #[cfg(target_endian = "little")]
                if bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0 {
                    (bytes_to_f32_slice(bytes), false)
                } else {
                    (
                        meta.converted_f32
                            .get_or_init(|| copy_bytes_to_f32_owned(bytes).into_boxed_slice())
                            .as_ref(),
                        false,
                    )
                }
                #[cfg(not(target_endian = "little"))]
                {
                    (
                        meta.converted_f32
                            .get_or_init(|| copy_bytes_to_f32_owned(bytes).into_boxed_slice())
                            .as_ref(),
                        false,
                    )
                }
            }
            // F16/BF16 -> f32 materialization is unconditional: the bit-conversion
            // below always compiles and adds no dependency. `f16` still gates F8
            // materialization (below) and the Cargo.toml targets that require it.
            DType::F16 => (
                meta.converted_f32
                    .get_or_init(|| {
                        let (values, has_non_finite) = convert_f16_bytes_to_f32(bytes);
                        let _ = meta.validated.get_or_init(|| {
                            let tensor = if has_non_finite {
                                crate::weights::ingress::IngestedTensor::decoded_f32(
                                    source, name, shape, dtype_name, &values,
                                )
                            } else {
                                crate::weights::ingress::IngestedTensor::decoded_f32_known_finite(
                                    source, name, shape, dtype_name, &values,
                                )
                            };
                            crate::weights::ingress::validate_ingested_tensor(tensor)
                                .map_err(|e| e.to_string())
                        });
                        values.into_boxed_slice()
                    })
                    .as_ref(),
                true,
            ),
            DType::BF16 => (
                meta.converted_f32
                    .get_or_init(|| {
                        let (values, has_non_finite) = convert_bf16_bytes_to_f32(bytes);
                        let _ = meta.validated.get_or_init(|| {
                            let tensor = if has_non_finite {
                                crate::weights::ingress::IngestedTensor::decoded_f32(
                                    source, name, shape, dtype_name, &values,
                                )
                            } else {
                                crate::weights::ingress::IngestedTensor::decoded_f32_known_finite(
                                    source, name, shape, dtype_name, &values,
                                )
                            };
                            crate::weights::ingress::validate_ingested_tensor(tensor)
                                .map_err(|e| e.to_string())
                        });
                        values.into_boxed_slice()
                    })
                    .as_ref(),
                true,
            ),
            DType::F8E4M3 => {
                #[cfg(feature = "f16")]
                {
                    (
                        meta.converted_f32
                            .get_or_init(|| {
                                let (values, has_non_finite) = convert_f8_e4m3_bytes_to_f32(bytes);
                                let _ = meta.validated.get_or_init(|| {
                                    let tensor = if has_non_finite {
                                        crate::weights::ingress::IngestedTensor::decoded_f32(
                                            source, name, shape, dtype_name, &values,
                                        )
                                    } else {
                                        crate::weights::ingress::IngestedTensor::decoded_f32_known_finite(
                                            source, name, shape, dtype_name, &values,
                                        )
                                    };
                                    crate::weights::ingress::validate_ingested_tensor(tensor)
                                        .map_err(|e| e.to_string())
                                });
                                values.into_boxed_slice()
                            })
                            .as_ref(),
                        true,
                    )
                }
                #[cfg(not(feature = "f16"))]
                {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "tensor {name} is F8_E4M3 but lattice-inference was built without the f16 feature"
                    )));
                }
            }
            DType::F8E5M2 => {
                #[cfg(feature = "f16")]
                {
                    (
                        meta.converted_f32
                            .get_or_init(|| {
                                let (values, has_non_finite) = convert_f8_e5m2_bytes_to_f32(bytes);
                                let _ = meta.validated.get_or_init(|| {
                                    let tensor = if has_non_finite {
                                        crate::weights::ingress::IngestedTensor::decoded_f32(
                                            source, name, shape, dtype_name, &values,
                                        )
                                    } else {
                                        crate::weights::ingress::IngestedTensor::decoded_f32_known_finite(
                                            source, name, shape, dtype_name, &values,
                                        )
                                    };
                                    crate::weights::ingress::validate_ingested_tensor(tensor)
                                        .map_err(|e| e.to_string())
                                });
                                values.into_boxed_slice()
                            })
                            .as_ref(),
                        true,
                    )
                }
                #[cfg(not(feature = "f16"))]
                {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "tensor {name} is F8_E5M2 but lattice-inference was built without the f16 feature"
                    )));
                }
            }
            DType::Other { label, .. } => {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name} has unsupported dtype {label} (source: {}); only F32, F16, \
                     BF16, F8_E4M3, and F8_E5M2 tensors can be materialized as f32",
                    self.source
                )));
            }
        };

        let validation = if validation_is_fused {
            meta.validated.get().ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "{source}: tensor {name} ({dtype_name}) widening completed without a \
                     validation result"
                ))
            })?
        } else {
            // `get_or_init` runs its closure at most once even under concurrent
            // first access, so zero-copy F32 pages are scanned by one caller.
            meta.validated.get_or_init(|| {
                crate::weights::ingress::validate_ingested_tensor(
                    crate::weights::ingress::IngestedTensor::decoded_f32(
                        source, name, shape, dtype_name, slice,
                    ),
                )
                .map_err(|e| e.to_string())
            })
        };
        match validation {
            Ok(()) => {}
            Err(msg) => return Err(InferenceError::InvalidSafetensors(msg.clone())),
        }

        Ok((slice, meta.shape.as_slice()))
    }

    /// **Unstable**: load BERT weights; tensor name conventions and signature may change.
    ///
    /// Load all BERT weights from the safetensors file.
    ///
    /// Encoder tensors may be flat or share one dot-delimited namespace.
    /// Mixed encoder namespaces are rejected, including misplaced pooler tensors.
    pub fn load_bert_weights(
        &self,
        num_layers: usize,
        hidden_size: usize,
    ) -> Result<BertWeights<'_>, InferenceError> {
        let encoder = BertTensorReader::new(self)?;
        let word_shape = encoder.require_shape("embeddings.word_embeddings.weight")?;
        let position_shape = encoder.require_shape("embeddings.position_embeddings.weight")?;
        let token_type_shape = encoder.require_shape("embeddings.token_type_embeddings.weight")?;

        if word_shape.len() != 2 || word_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: encoder
                    .name("embeddings.word_embeddings.weight")
                    .into_owned(),
                expected: vec![word_shape[0], hidden_size],
                actual: word_shape.to_vec(),
            });
        }
        if position_shape.len() != 2 || position_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: encoder
                    .name("embeddings.position_embeddings.weight")
                    .into_owned(),
                expected: vec![position_shape[0], hidden_size],
                actual: position_shape.to_vec(),
            });
        }
        if token_type_shape.len() != 2 || token_type_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: encoder
                    .name("embeddings.token_type_embeddings.weight")
                    .into_owned(),
                expected: vec![token_type_shape[0], hidden_size],
                actual: token_type_shape.to_vec(),
            });
        }

        let word_embeddings = encoder.tensor2d(
            "embeddings.word_embeddings.weight",
            word_shape[0],
            word_shape[1],
        )?;
        let position_embeddings = encoder.tensor2d(
            "embeddings.position_embeddings.weight",
            position_shape[0],
            position_shape[1],
        )?;
        let token_type_embeddings = encoder.tensor2d(
            "embeddings.token_type_embeddings.weight",
            token_type_shape[0],
            token_type_shape[1],
        )?;
        let embedding_layer_norm_weight =
            encoder.tensor1d("embeddings.LayerNorm.weight", hidden_size)?;
        let embedding_layer_norm_bias =
            encoder.tensor1d("embeddings.LayerNorm.bias", hidden_size)?;

        let intermediate_shape =
            encoder.require_shape("encoder.layer.0.intermediate.dense.weight")?;
        if intermediate_shape.len() != 2 || intermediate_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: encoder
                    .name("encoder.layer.0.intermediate.dense.weight")
                    .into_owned(),
                expected: vec![intermediate_shape[0], hidden_size],
                actual: intermediate_shape.to_vec(),
            });
        }
        let intermediate_size = intermediate_shape[0];

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("encoder.layer.{i}");
            layers.push(TransformerLayerWeights {
                query_weight: encoder.tensor2d(
                    &format!("{prefix}.attention.self.query.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                query_bias: encoder
                    .tensor1d(&format!("{prefix}.attention.self.query.bias"), hidden_size)?,
                key_weight: encoder.tensor2d(
                    &format!("{prefix}.attention.self.key.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                key_bias: encoder
                    .tensor1d(&format!("{prefix}.attention.self.key.bias"), hidden_size)?,
                value_weight: encoder.tensor2d(
                    &format!("{prefix}.attention.self.value.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                value_bias: encoder
                    .tensor1d(&format!("{prefix}.attention.self.value.bias"), hidden_size)?,
                attn_output_weight: encoder.tensor2d(
                    &format!("{prefix}.attention.output.dense.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                attn_output_bias: encoder.tensor1d(
                    &format!("{prefix}.attention.output.dense.bias"),
                    hidden_size,
                )?,
                attn_layer_norm_weight: encoder.tensor1d(
                    &format!("{prefix}.attention.output.LayerNorm.weight"),
                    hidden_size,
                )?,
                attn_layer_norm_bias: encoder.tensor1d(
                    &format!("{prefix}.attention.output.LayerNorm.bias"),
                    hidden_size,
                )?,
                ffn_intermediate_weight: encoder.tensor2d(
                    &format!("{prefix}.intermediate.dense.weight"),
                    intermediate_size,
                    hidden_size,
                )?,
                ffn_intermediate_bias: encoder.tensor1d(
                    &format!("{prefix}.intermediate.dense.bias"),
                    intermediate_size,
                )?,
                ffn_output_weight: encoder.tensor2d(
                    &format!("{prefix}.output.dense.weight"),
                    hidden_size,
                    intermediate_size,
                )?,
                ffn_output_bias: encoder
                    .tensor1d(&format!("{prefix}.output.dense.bias"), hidden_size)?,
                ffn_layer_norm_weight: encoder
                    .tensor1d(&format!("{prefix}.output.LayerNorm.weight"), hidden_size)?,
                ffn_layer_norm_bias: encoder
                    .tensor1d(&format!("{prefix}.output.LayerNorm.bias"), hidden_size)?,
            });
        }

        let pooler_weight =
            encoder.tensor2d_optional("pooler.dense.weight", hidden_size, hidden_size)?;
        let pooler_bias = encoder.tensor1d_optional("pooler.dense.bias", hidden_size)?;

        Ok(BertWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_layer_norm_weight,
            embedding_layer_norm_bias,
            layers,
            pooler_weight,
            pooler_bias,
        })
    }

    /// Load classifier head weights from a `BertForSequenceClassification` safetensors file.
    ///
    /// Accepts `classifier.weight` shaped `[1, hidden_size]` or `[hidden_size]` and
    /// `classifier.bias` shaped `[1]`. Classifier tensors must remain at the top
    /// level, independently of the BERT encoder namespace.
    pub fn load_cross_encoder_weights(
        &self,
        hidden_size: usize,
    ) -> Result<CrossEncoderWeights, InferenceError> {
        let (weight, weight_shape) = self.get_f32_tensor("classifier.weight")?;
        let classifier_weight = match weight_shape {
            [1, cols] if *cols == hidden_size => weight.to_vec(),
            [cols] if *cols == hidden_size => weight.to_vec(),
            _ => {
                return Err(InferenceError::ShapeMismatch {
                    name: "classifier.weight".to_string(),
                    expected: vec![1, hidden_size],
                    actual: weight_shape.to_vec(),
                });
            }
        };

        let (bias, bias_shape) = self.get_f32_tensor("classifier.bias")?;
        if bias_shape != [1usize] {
            return Err(InferenceError::ShapeMismatch {
                name: "classifier.bias".to_string(),
                expected: vec![1],
                actual: bias_shape.to_vec(),
            });
        }
        let classifier_bias = bias[0];

        Ok(CrossEncoderWeights {
            classifier_weight,
            classifier_bias,
        })
    }

    fn require_shape(&self, name: &str) -> Result<&[usize], InferenceError> {
        self.tensor_shape(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    fn tensor1d(&self, name: &str, len: usize) -> Result<Tensor1D<'_>, InferenceError> {
        let (data, shape) = self.get_f32_tensor(name)?;
        if shape != [len] {
            return Err(InferenceError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![len],
                actual: shape.to_vec(),
            });
        }
        Ok(Tensor1D { data, len })
    }

    fn tensor2d(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'_>, InferenceError> {
        let (data, shape) = self.get_f32_tensor(name)?;
        if shape != [rows, cols] {
            return Err(InferenceError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![rows, cols],
                actual: shape.to_vec(),
            });
        }
        Ok(Tensor2D { data, rows, cols })
    }

    fn tensor1d_optional(&self, name: &str, len: usize) -> Result<Tensor1D<'_>, InferenceError> {
        if self.has_tensor(name) {
            self.tensor1d(name, len)
        } else {
            Ok(Tensor1D { data: &[], len: 0 })
        }
    }

    fn tensor2d_optional(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'_>, InferenceError> {
        if self.has_tensor(name) {
            self.tensor2d(name, rows, cols)
        } else {
            Ok(Tensor2D {
                data: &[],
                rows: 0,
                cols: 0,
            })
        }
    }
}

struct BertTensorReader<'a> {
    file: &'a SafetensorsFile,
    prefix: &'a str,
}

impl<'a> BertTensorReader<'a> {
    fn new(file: &'a SafetensorsFile) -> Result<Self, InferenceError> {
        let mut prefixes = BTreeSet::new();
        for name in file.tensors.keys() {
            // Head shadows cannot establish an encoder namespace.
            if name.ends_with(".classifier.weight") || name.ends_with(".classifier.bias") {
                continue;
            }
            let offset = ["embeddings.", "encoder.layer.", "pooler."]
                .into_iter()
                .flat_map(|family| name.match_indices(family))
                .map(|(offset, _)| offset)
                .filter(|&offset| offset == 0 || name.as_bytes()[offset - 1] == b'.')
                .max();
            if let Some(offset) = offset {
                prefixes.insert(&name[..offset]);
            }
        }
        if prefixes.len() > 1 {
            return Err(InferenceError::InvalidSafetensors(format!(
                "mixed BERT encoder namespaces: {prefixes:?} (empty prefix is unprefixed)"
            )));
        }
        let prefix = prefixes.into_iter().next().unwrap_or("");
        Ok(Self { file, prefix })
    }

    fn name<'n>(&self, name: &'n str) -> Cow<'n, str> {
        if self.prefix.is_empty() {
            Cow::Borrowed(name)
        } else {
            Cow::Owned(format!("{}{name}", self.prefix))
        }
    }

    fn require_shape(&self, name: &str) -> Result<&'a [usize], InferenceError> {
        self.file.require_shape(&self.name(name))
    }

    fn tensor1d(&self, name: &str, len: usize) -> Result<Tensor1D<'a>, InferenceError> {
        self.file.tensor1d(&self.name(name), len)
    }

    fn tensor2d(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'a>, InferenceError> {
        self.file.tensor2d(&self.name(name), rows, cols)
    }

    fn tensor1d_optional(&self, name: &str, len: usize) -> Result<Tensor1D<'a>, InferenceError> {
        self.file.tensor1d_optional(&self.name(name), len)
    }

    fn tensor2d_optional(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'a>, InferenceError> {
        self.file.tensor2d_optional(&self.name(name), rows, cols)
    }
}

#[cfg(test)]
mod bert_namespace_tests;

/// **Unstable**: per-layer Qwen3 decoder weights; field set may change with model variants.
///
/// Weights for one Qwen3 decoder layer (GQA + SwiGLU + QK-norm).
#[derive(Debug, Clone)]
pub struct QwenLayerWeights<'a> {
    // Self-attention (GQA: Q has num_heads, K/V have num_kv_heads)
    pub q_proj_weight: Tensor2D<'a>,
    pub k_proj_weight: Tensor2D<'a>,
    pub v_proj_weight: Tensor2D<'a>,
    pub o_proj_weight: Tensor2D<'a>,
    // QK normalization (per-head RMSNorm, shape [head_dim])
    pub q_norm_weight: Tensor1D<'a>,
    pub k_norm_weight: Tensor1D<'a>,
    // Pre-attention RMSNorm
    pub input_layernorm_weight: Tensor1D<'a>,
    // SwiGLU FFN (no bias)
    pub gate_proj_weight: Tensor2D<'a>,
    pub up_proj_weight: Tensor2D<'a>,
    pub down_proj_weight: Tensor2D<'a>,
    // Pre-FFN RMSNorm
    pub post_attention_layernorm_weight: Tensor1D<'a>,

    /// Fused QKV weight: [q_dim + 2*kv_dim, hidden_size] — rows of W_q, W_k, W_v concatenated.
    /// Enables a single GEMM call instead of 3 separate projections.
    pub fused_qkv: Vec<f32>,
    /// Output dimension of fused QKV: q_dim + 2 * kv_dim.
    pub qkv_out_dim: usize,

    /// Fused gate+up weight: [2*intermediate_size, hidden_size] — rows of W_gate, W_up concatenated.
    /// Enables a single GEMM call instead of 2 separate projections.
    pub fused_gate_up: Vec<f32>,
    /// Output dimension of fused gate+up: 2 * intermediate_size.
    pub gate_up_out_dim: usize,
}

/// **Unstable**: full Qwen3 model weight bundle; structure may change with model variants.
///
/// All weights for a Qwen3 model.
#[derive(Debug, Clone)]
pub struct QwenWeights<'a> {
    pub embed_tokens: Tensor2D<'a>,
    pub norm_weight: Tensor1D<'a>,
    pub layers: Vec<QwenLayerWeights<'a>>,
}

impl SafetensorsFile {
    /// **Unstable**: load Qwen3 weights; tensor name conventions and fused-weight strategy may change.
    ///
    /// Load all Qwen3 weights from the safetensors file.
    pub fn load_qwen_weights(
        &self,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Result<QwenWeights<'_>, InferenceError> {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let embed_shape = self.require_shape("embed_tokens.weight")?;
        if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "embed_tokens.weight".into(),
                expected: vec![embed_shape[0], hidden_size],
                actual: embed_shape.to_vec(),
            });
        }

        let embed_tokens = self.tensor2d("embed_tokens.weight", embed_shape[0], hidden_size)?;
        let norm_weight = self.tensor1d("norm.weight", hidden_size)?;

        let qkv_out_dim = q_dim + 2 * kv_dim;
        let gate_up_out_dim = 2 * intermediate_size;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let p = format!("layers.{i}");

            // Load individual weight tensors first.
            let q_proj_weight =
                self.tensor2d(&format!("{p}.self_attn.q_proj.weight"), q_dim, hidden_size)?;
            let k_proj_weight =
                self.tensor2d(&format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden_size)?;
            let v_proj_weight =
                self.tensor2d(&format!("{p}.self_attn.v_proj.weight"), kv_dim, hidden_size)?;
            let gate_proj_weight = self.tensor2d(
                &format!("{p}.mlp.gate_proj.weight"),
                intermediate_size,
                hidden_size,
            )?;
            let up_proj_weight = self.tensor2d(
                &format!("{p}.mlp.up_proj.weight"),
                intermediate_size,
                hidden_size,
            )?;

            // Build fused QKV weight: [q_dim + 2*kv_dim, hidden_size]
            // Rows of W_q, W_k, W_v concatenated vertically.
            let mut fused_qkv = Vec::with_capacity(qkv_out_dim * hidden_size);
            fused_qkv.extend_from_slice(q_proj_weight.data); // [q_dim * hidden_size]
            fused_qkv.extend_from_slice(k_proj_weight.data); // [kv_dim * hidden_size]
            fused_qkv.extend_from_slice(v_proj_weight.data); // [kv_dim * hidden_size]

            // Build fused gate+up weight: [2*intermediate_size, hidden_size]
            // Rows of W_gate, W_up concatenated vertically.
            let mut fused_gate_up = Vec::with_capacity(gate_up_out_dim * hidden_size);
            fused_gate_up.extend_from_slice(gate_proj_weight.data); // [intermediate * hidden]
            fused_gate_up.extend_from_slice(up_proj_weight.data); // [intermediate * hidden]

            layers.push(QwenLayerWeights {
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                o_proj_weight: self.tensor2d(
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden_size,
                    q_dim,
                )?,
                q_norm_weight: self.tensor1d(&format!("{p}.self_attn.q_norm.weight"), head_dim)?,
                k_norm_weight: self.tensor1d(&format!("{p}.self_attn.k_norm.weight"), head_dim)?,
                input_layernorm_weight: self
                    .tensor1d(&format!("{p}.input_layernorm.weight"), hidden_size)?,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight: self.tensor2d(
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden_size,
                    intermediate_size,
                )?,
                post_attention_layernorm_weight: self
                    .tensor1d(&format!("{p}.post_attention_layernorm.weight"), hidden_size)?,
                fused_qkv,
                qkv_out_dim,
                fused_gate_up,
                gate_up_out_dim,
            });
        }

        Ok(QwenWeights {
            embed_tokens,
            norm_weight,
            layers,
        })
    }
}

/// Reinterpret a byte slice as a f32 slice (zero-copy).
fn bytes_to_f32_slice(bytes: &[u8]) -> &[f32] {
    assert!(bytes.len().is_multiple_of(4));
    assert!(bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0);
    // SAFETY: The caller guarantees that `bytes` is 4-byte aligned, its length is
    // a multiple of 4, and the backing storage outlives the returned slice.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

fn copy_bytes_to_f32_owned(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0);
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn validate_bf16_payload(
    source: &str,
    name: &str,
    shape: &[usize],
    bytes: &[u8],
) -> Result<(), InferenceError> {
    let validator = crate::weights::ingress::DecodedTensorValidator::safetensors(
        source,
        name,
        shape,
        DType::BF16.name(),
        bytes.len() / 2,
    )?;
    let bits_at = |chunk: &[u8]| u16::from_le_bytes([chunk[0], chunk[1]]);
    let has_non_finite = bytes.chunks_exact(2).fold(false, |seen, chunk| {
        seen | (bits_at(chunk) & 0x7f80 == 0x7f80)
    });
    if has_non_finite
        && let Some((index, bits)) = bytes
            .chunks_exact(2)
            .map(bits_at)
            .enumerate()
            .find(|(_, bits)| bits & 0x7f80 == 0x7f80)
    {
        return Err(validator.reject_bf16_bits_at(index, bits));
    }
    Ok(())
}

fn convert_f16_bytes_to_f32(bytes: &[u8]) -> (Vec<f32>, bool) {
    debug_assert_eq!(bytes.len() % 2, 0);
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut has_non_finite = false;
    for chunk in bytes.chunks_exact(2) {
        let value =
            crate::weights::half_bits::f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        has_non_finite |= !value.is_finite();
        out.push(value);
    }
    (out, has_non_finite)
}

fn convert_bf16_bytes_to_f32(bytes: &[u8]) -> (Vec<f32>, bool) {
    debug_assert_eq!(bytes.len() % 2, 0);
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut has_non_finite = false;
    for chunk in bytes.chunks_exact(2) {
        let value =
            crate::weights::half_bits::bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        has_non_finite |= !value.is_finite();
        out.push(value);
    }
    (out, has_non_finite)
}

#[cfg(feature = "f16")]
fn convert_f8_e4m3_bytes_to_f32(bytes: &[u8]) -> (Vec<f32>, bool) {
    let mut out = Vec::with_capacity(bytes.len());
    let mut has_non_finite = false;
    for &byte in bytes {
        let value = crate::weights::half_bits::f8_e4m3_bits_to_f32(byte);
        has_non_finite |= !value.is_finite();
        out.push(value);
    }
    (out, has_non_finite)
}

#[cfg(feature = "f16")]
fn convert_f8_e5m2_bytes_to_f32(bytes: &[u8]) -> (Vec<f32>, bool) {
    let mut out = Vec::with_capacity(bytes.len());
    let mut has_non_finite = false;
    for &byte in bytes {
        let value = crate::weights::half_bits::f8_e5m2_bits_to_f32(byte);
        has_non_finite |= !value.is_finite();
        out.push(value);
    }
    (out, has_non_finite)
}

/// Owned backing store for Qwen weights loaded from a sharded checkpoint.
///
/// Each field is the raw `Vec<f32>` that a `Tensor2D` or `Tensor1D` slice
/// would point into.  The struct is heap-allocated and pinned behind a `Box`
/// so the slices remain valid for the lifetime of the `QwenWeights<'_>` that
/// borrows from it.
#[derive(Debug)]
pub struct ShardedQwenBacking {
    pub embed_tokens: Vec<f32>,
    pub norm_weight: Vec<f32>,
    // Per-layer fields; indexed by layer index.
    pub q_proj: Vec<Vec<f32>>,
    pub k_proj: Vec<Vec<f32>>,
    pub v_proj: Vec<Vec<f32>>,
    pub o_proj: Vec<Vec<f32>>,
    pub q_norm: Vec<Vec<f32>>,
    pub k_norm: Vec<Vec<f32>>,
    pub input_ln: Vec<Vec<f32>>,
    pub gate_proj: Vec<Vec<f32>>,
    pub up_proj: Vec<Vec<f32>>,
    pub down_proj: Vec<Vec<f32>>,
    pub post_ln: Vec<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Sharded safetensors index + TensorSource trait
// ---------------------------------------------------------------------------

/// Upper bound, in bytes, on `model.safetensors.index.json` accepted from a checkpoint
/// directory (FIX 6).
///
/// The index is opened once, checked for regular-file identity and initial size on that
/// handle, and read through the same handle with a one-byte overrun check. Real sharded
/// Qwen3.5/3.6 checkpoints (up to ~26 shards, tens of thousands of tensor names) produce
/// index files on the order of a few MiB; 64 MiB (67,108,864) leaves roughly an order of
/// magnitude of headroom while rejecting an oversized `weight_map` before parsing.
pub(crate) const MAX_SAFETENSORS_INDEX_BYTES: u64 = 67_108_864;

/// Upper bound on the number of entries in `SafetensorsIndex::weight_map` accepted from a
/// checkpoint directory (FIX 6).
///
/// Enforced incrementally inside [`deserialize_weight_map_no_duplicates`]'s visitor --
/// admission fires before a further entry is inserted into the owned `HashMap`, not after
/// the whole map has already collapsed. Real Qwen3.5/3.6 sharded checkpoints (including the
/// 256-expert MoE 35B preset) carry on the order of tens of thousands of tensor names;
/// 1,000,000 leaves well over an order of magnitude of headroom while rejecting an
/// unbounded entry count.
pub(crate) const MAX_WEIGHT_MAP_ENTRIES: usize = 1_000_000;

/// Upper bound, in bytes, on the safetensors header (`header_len` at the start of a
/// `.safetensors` file, checked in [`SafetensorsFile::from_backing`] before the header
/// is parsed as JSON).
///
/// An unsharded checkpoint puts metadata for every tensor in this one header, so it can
/// run larger than a single shard's slice of `model.safetensors.index.json`. Real dense
/// and MoE checkpoints, including the largest single-file presets this crate loads,
/// produce headers on the order of a few hundred KiB to low single-digit MiB, so
/// compatibility alone would tolerate a much larger cap.
///
/// This value is set by a second, tighter bound: what the parser retains after
/// `open()` returns, not what a real header looks like. Every tensor's `shape` is
/// stored as a `Vec<usize>` (8 bytes/element on this crate's supported targets), and
/// [`validate_safetensors_layout`]'s element-count product accepts a zero-length
/// dimension (`checked_mul` by 0 is always in range), so a tensor entry needs no
/// corresponding tensor-data bytes once any one of its `shape` entries is `0`. A
/// header can therefore spend nearly its whole byte budget on a single tensor's
/// `shape` array of 8-byte elements written as one-digit decimals (`"0,"` is 2 input
/// bytes per retained element). Measured `Vec<usize>` growth under repeated `push`
/// (this crate's `parse_usize_array` never calls `with_capacity`) lands on the next
/// power of two at or above the element count, so at a power-of-two byte cap this
/// construction retains almost exactly 4x the cap: 4 MiB of header bounds worst-case
/// retained shape-array bytes to 16 MiB. (Tensor names, `TensorMeta`'s fixed fields,
/// and the `HashMap` bucket overhead were checked too: at 120 bytes/entry plus a
/// short name, spreading the same byte budget across many small tensor entries
/// instead retains less per input byte than the single-oversized-shape
/// construction, so they do not set the bound.)
///
/// This sizing assumes the ordinary Rust allocator contract: growing a `Vec` past
/// available memory via plain `push` is not a `Result` calling code can recover
/// from, so the input-side byte cap is what keeps retained parser state bounded,
/// not error handling downstream of the allocation.
pub(crate) const MAX_SAFETENSORS_HEADER_BYTES: usize = 4_194_304;

/// Crate-private compatibility ceiling for dormant prepared framing checks.
pub(crate) const fn max_safetensors_header_bytes() -> usize {
    MAX_SAFETENSORS_HEADER_BYTES
}

/// Upper bound on JSON container nesting depth (`{` / `[`) while parsing a safetensors
/// header, tracked by [`JsonParser::depth`] and enforced in [`JsonParser::skip_value`].
///
/// A real header is shallow: the top-level object holds per-tensor objects, each holding
/// at most a `shape`/`data_offsets` array — three levels deep, four counting `__metadata__`
/// values written by common tooling. 32 leaves an order of magnitude of headroom over that.
const MAX_SAFETENSORS_HEADER_DEPTH: usize = 32;

/// Parsed `model.safetensors.index.json` from a HuggingFace sharded checkpoint.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SafetensorsIndex {
    #[serde(default)]
    pub metadata: Value,
    /// Maps tensor name → shard filename (e.g. `"model-00001-of-00026.safetensors"`).
    #[serde(deserialize_with = "deserialize_weight_map_no_duplicates")]
    pub weight_map: HashMap<String, String>,
}

/// Deserializes `weight_map` from the raw JSON object, rejecting a repeated
/// tensor-name key before it collapses into the `HashMap`.
///
/// A raw JSON object can carry the same key twice (e.g. mapped to two
/// different shard files); ordinary map deserialization silently keeps
/// whichever value visits last, so a downstream inventory-count check over
/// the resulting `HashMap` can never observe the duplicate. Counting raw
/// member visits here, before collapse, closes that gap.
fn deserialize_weight_map_no_duplicates<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct WeightMapVisitor;

    impl<'de> serde::de::Visitor<'de> for WeightMapVisitor {
        type Value = HashMap<String, String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a map of tensor name to shard filename")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            // Cap the initial reservation independent of the (attacker-controlled)
            // declared `size_hint`, then reject before a further entry is inserted once
            // over budget -- bounding growth during the parse itself rather than after
            // the map has already fully materialized. See `MAX_WEIGHT_MAP_ENTRIES` docs.
            let reserve = map.size_hint().unwrap_or(0).min(MAX_WEIGHT_MAP_ENTRIES);
            let mut out = HashMap::with_capacity(reserve);
            while let Some((name, shard)) = map.next_entry::<String, String>()? {
                if out.len() >= MAX_WEIGHT_MAP_ENTRIES {
                    return Err(serde::de::Error::custom(format!(
                        "weight_map exceeds MAX_WEIGHT_MAP_ENTRIES ({MAX_WEIGHT_MAP_ENTRIES})"
                    )));
                }
                if out.insert(name.clone(), shard).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate tensor name in weight_map: {name}"
                    )));
                }
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(WeightMapVisitor)
}

/// Owned tensor with f32 data and shape, returned by `load_sharded`.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Uniform interface for single-file and sharded tensor sources.
pub trait TensorSource {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError>;
    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError>;
    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError>;
    /// Header-declared dtype name for a named tensor, `None` if the source
    /// cannot report it. Default `None` keeps every existing implementor
    /// (mocks, sharded sources that haven't wired this through yet) source
    /// compatible; a loader that needs a dtype contract enforced treats
    /// `None` as "unknown, cannot check" rather than a failure.
    fn tensor_dtype(&mut self, _name: &str) -> Result<Option<String>, InferenceError> {
        Ok(None)
    }
}

impl TensorSource for SafetensorsFile {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        Ok(SafetensorsFile::has_tensor(self, name))
    }

    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        Ok(SafetensorsFile::tensor_shape(self, name).map(<[usize]>::to_vec))
    }

    fn tensor_dtype(&mut self, name: &str) -> Result<Option<String>, InferenceError> {
        Ok(SafetensorsFile::tensor_dtype(self, name).map(str::to_string))
    }

    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        let shape = self.get_f32_tensor(name)?.1.to_vec();
        let meta = self
            .tensors
            .get_mut(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;
        // Handing the cached conversion over instead of cloning it keeps one f32 copy
        // of the tensor alive during a load, not two. Only the zero-copy aligned F32
        // view has nothing cached; that case still needs the copy.
        if let Some(converted) = meta.converted_f32.take() {
            return Ok((converted.into_vec(), shape));
        }
        let (data, _) = self.get_f32_tensor(name)?;
        Ok((data.to_vec(), shape))
    }
}

/// Lazy-opening reader for sharded safetensors checkpoints.
///
/// Opens shard files on demand (first access) to avoid mapping 26 files upfront.
#[derive(Debug)]
pub struct ShardedSafetensors {
    root: PathBuf,
    index: SafetensorsIndex,
    shards: HashMap<String, SafetensorsFile>,
}

/// Resolve an index-declared shard file name beneath `model_dir`, rejecting
/// entries that escape the model directory.
///
/// Shard names come from checkpoint-controlled JSON (`model.safetensors.
/// index.json` or a quantized-checkpoint manifest), so they are untrusted
/// input: an absolute entry replaces `model_dir` entirely in `Path::join`,
/// and `..` components address files outside the checkpoint — turning a
/// malicious model directory into an open-arbitrary-readable-file
/// primitive (#1069). The validation is purely lexical — absolute paths
/// and any parent-directory (or Windows prefix/root) component are
/// rejected before the join — so an index entry cannot address anything
/// above the model directory, and no filesystem state is consulted,
/// leaving no check-to-open window to race.
///
/// # Threat model
///
/// Manifest STRINGS are untrusted; the local FILESYSTEM is trusted.
/// Symlinks are deliberately followed at open time, matching peer loaders
/// (HF transformers, candle, llama.cpp). The HuggingFace hub cache stores
/// snapshots as symlink farms (`snapshots/<rev>/x.safetensors ->
/// ../../blobs/<sha>`) and lattice loads those directories directly;
/// resolving symlinks and requiring the resolved target to stay beneath
/// the model directory — or refusing to follow symlinks at all — would
/// reject every hub-cache checkpoint (pinned by
/// `sharded_loader_follows_hub_cache_snapshot_layout`, which fails under
/// exactly such a policy). A symlink planted inside the model directory
/// is therefore honored by design: an actor who can write the model
/// directory already controls every byte the loader reads, so blocking
/// links there defends nothing, and archive-extraction tools are the
/// correct place to refuse hostile symlinks. Do not "re-harden" this
/// helper to resolve or reject symlinks without revisiting that decision
/// trail (#1069, #1072).
pub fn contained_shard_path(model_dir: &Path, shard_file: &str) -> Result<PathBuf, InferenceError> {
    let rel = Path::new(shard_file);
    if rel.as_os_str().is_empty() {
        return Err(InferenceError::InvalidSafetensors(
            "shard entry is empty; index entries must name a file within the model directory"
                .to_string(),
        ));
    }
    if rel.is_absolute() {
        return Err(InferenceError::InvalidSafetensors(format!(
            "shard entry {shard_file:?} is an absolute path; \
             index entries must stay within the model directory"
        )));
    }
    for component in rel.components() {
        match component {
            std::path::Component::Normal(_) | std::path::Component::CurDir => {}
            _ => {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "shard entry {shard_file:?} escapes the model directory; \
                     index entries must stay within the model directory"
                )));
            }
        }
    }
    Ok(model_dir.join(rel))
}

/// Parse `model.safetensors.index.json` from a model directory.
pub fn parse_index(model_dir: &Path) -> Result<SafetensorsIndex, InferenceError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    // The shared reader checks the open handle and bounds its read to the same handle.
    let json =
        read_text_bounded(&index_path, MAX_SAFETENSORS_INDEX_BYTES).map_err(
            |error| match error {
                BoundedReadError::TooLarge { len, cap } => {
                    InferenceError::InvalidSafetensors(format!(
                        "{} is {len} bytes, exceeding MAX_SAFETENSORS_INDEX_BYTES \
                         ({cap})",
                        index_path.display()
                    ))
                }
                BoundedReadError::NotRegularFile => InferenceError::InvalidSafetensors(format!(
                    "{} is not a regular file",
                    index_path.display()
                )),
                BoundedReadError::Io(error) => InferenceError::Io(error),
            },
        )?;
    serde_json::from_str(&json).map_err(|e| {
        InferenceError::InvalidSafetensors(format!("failed to parse {}: {e}", index_path.display()))
    })
}

/// Open a manifest-derived shard entry exactly once, returning the open [`File`] and the
/// real on-disk path it resolved to.
///
/// `entry_name` comes from an untrusted checkpoint manifest (`model.safetensors.index.json`'s
/// `weight_map`, or `quantize_index.json`'s per-tensor `file` field), so it is validated by
/// [`contained_shard_path`] before the join: absolute paths and parent-directory components
/// are rejected, lexically, without consulting any filesystem state.
///
/// The returned file has already passed
/// [`crate::weights::mmap_trust::reject_if_open_mmap_file_untrusted`] against its resolved
/// real path -- every caller mmaps this file read-only no-copy, so the same write-boundary
/// gate every other checkpoint mmap goes through applies here too, without `open_trusted_mmap_file`'s
/// `O_NOFOLLOW` (see that function's doc comment for why a hub-cache symlink must still resolve).
///
/// # Why this opens the file instead of returning a path to reopen
///
/// Resolving a path and handing it back leaves the caller to `open()` it separately, and a
/// concurrent writer can swap the file or symlink in between: the path that was checked and
/// the path that was opened are then not the same file. This function opens the candidate
/// first and reports the identity of the descriptor actually opened. **Never reopen the path
/// this returns** — read from the returned `File`. The returned path is for diagnostics and
/// error text only, and nothing security-relevant keys off it.
///
/// # What this deliberately does NOT check
///
/// It does not require the opened file's real path to stay beneath the model root, and it
/// follows symlinks. That matches the threat model documented on [`contained_shard_path`]:
/// manifest strings are untrusted, the local filesystem is trusted. Asserting containment on
/// the *resolved* path would reject every HuggingFace hub-cache checkpoint, since those
/// snapshots are symlink farms (`snapshots/<rev>/x.safetensors -> ../../blobs/<sha>`) and
/// lattice loads such directories directly. Read that threat-model section before
/// reintroducing a resolved-path containment check here.
pub(crate) fn open_manifest_entry_once(
    model_root: &Path,
    entry_name: &str,
) -> Result<(File, PathBuf), InferenceError> {
    let candidate = contained_shard_path(model_root, entry_name)?;
    let file = crate::weights::mmap_trust::open_regular_file_no_hang(&candidate)
        .map_err(InferenceError::InvalidSafetensors)?;
    let real_path = real_path_of_open_file(&file, &candidate)?;
    crate::weights::mmap_trust::reject_if_open_mmap_file_untrusted(&file, &real_path)
        .map_err(InferenceError::InvalidSafetensors)?;
    Ok((file, real_path))
}

/// Resolve the real on-disk path of an already-open file descriptor.
///
/// On macOS, uses `fcntl(fd, F_GETPATH, buf)` -- the fd-identity source of truth, immune
/// to a path swap that happens after `open()` returns. On other platforms (no portable
/// fd-to-path syscall in `std`), falls back to canonicalizing the path that was passed to
/// `open()`; this is a narrower, best-effort check that does not close the TOCTOU window
/// as completely as the macOS path, matching this crate's macOS-primary target (Metal GPU,
/// e2e-parity CI runners).
#[cfg(target_os = "macos")]
fn real_path_of_open_file(file: &File, _candidate: &Path) -> Result<PathBuf, InferenceError> {
    use std::os::unix::io::AsRawFd;
    let mut buf = [0u8; libc::PATH_MAX as usize];
    let ret = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETPATH, buf.as_mut_ptr()) };
    if ret == -1 {
        return Err(InferenceError::Io(std::io::Error::last_os_error()));
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let s = std::str::from_utf8(&buf[..len]).map_err(|_| {
        InferenceError::Inference("F_GETPATH returned a non-UTF-8 path".to_string())
    })?;
    Ok(PathBuf::from(s))
}

#[cfg(not(target_os = "macos"))]
fn real_path_of_open_file(_file: &File, candidate: &Path) -> Result<PathBuf, InferenceError> {
    candidate.canonicalize().map_err(InferenceError::Io)
}

/// Resolve a tensor name to its shard path, contained within `model_dir`.
pub fn resolve_shard(
    index: &SafetensorsIndex,
    model_dir: &Path,
    tensor_name: &str,
) -> Result<PathBuf, InferenceError> {
    let shard_file = index
        .weight_map
        .get(tensor_name)
        .ok_or_else(|| InferenceError::MissingTensor(tensor_name.to_string()))?;
    contained_shard_path(model_dir, shard_file)
}

/// Eagerly load all tensors from a sharded checkpoint into an owned map.
///
/// Groups tensor names by shard file to open each shard file exactly once.
/// For large checkpoints (e.g. 26-shard Qwen3.6), prefer `ShardedSafetensors`
/// for lower peak memory during model loading.
pub fn load_sharded(model_dir: &Path) -> Result<HashMap<String, Tensor>, InferenceError> {
    let index = parse_index(model_dir)?;
    let mut by_shard: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();

    for (tensor_name, shard_file) in &index.weight_map {
        by_shard
            .entry(shard_file.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    let mut tensors = HashMap::with_capacity(index.weight_map.len());
    for (shard_file, tensor_names) in by_shard {
        let (file, real_path) = open_manifest_entry_once(model_dir, &shard_file)?;
        let shard = SafetensorsFile::from_open_file(file, &real_path)?;
        for tensor_name in tensor_names {
            let (data, shape) = shard.get_f32_tensor(&tensor_name)?;
            tensors.insert(
                tensor_name,
                Tensor {
                    data: data.to_vec(),
                    shape: shape.to_vec(),
                },
            );
        }
    }

    Ok(tensors)
}

impl ShardedSafetensors {
    /// Parse `model.safetensors.index.json` and set up the lazy shard reader.
    pub fn open_index(index_path: &Path) -> Result<Self, InferenceError> {
        let root = index_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let index = parse_index(&root)?;
        Ok(Self {
            root,
            index,
            shards: HashMap::new(),
        })
    }

    /// Return a reference to the parsed index.
    pub fn index(&self) -> &SafetensorsIndex {
        &self.index
    }

    /// Resolve a tensor name to `(shard_path, tensor_name)`, contained within `self.root`.
    pub fn resolve_weight(&self, tensor_name: &str) -> Result<(PathBuf, String), InferenceError> {
        let shard_file = self
            .index
            .weight_map
            .get(tensor_name)
            .ok_or_else(|| InferenceError::MissingTensor(tensor_name.to_string()))?;
        let shard_path = contained_shard_path(&self.root, shard_file)?;
        Ok((shard_path, tensor_name.to_string()))
    }

    fn shard_file_for(&self, name: &str) -> Result<String, InferenceError> {
        self.index
            .weight_map
            .get(name)
            .cloned()
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    fn open_shard(&mut self, shard_file: &str) -> Result<&mut SafetensorsFile, InferenceError> {
        if !self.shards.contains_key(shard_file) {
            let (file, real_path) = open_manifest_entry_once(&self.root, shard_file)?;
            let shard = SafetensorsFile::from_open_file(file, &real_path)?;
            self.shards.insert(shard_file.to_string(), shard);
        }
        self.shards.get_mut(shard_file).ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!("failed to cache shard {shard_file}"))
        })
    }
}

impl TensorSource for ShardedSafetensors {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        Ok(self.index.weight_map.contains_key(name))
    }

    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        let shard_file = self.shard_file_for(name)?;
        // Shared reborrow: a `&mut` receiver would resolve `tensor_shape` to this trait's method.
        let shard: &SafetensorsFile = self.open_shard(&shard_file)?;
        Ok(shard.tensor_shape(name).map(<[usize]>::to_vec))
    }

    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        let shard_file = self.shard_file_for(name)?;
        let shard = self.open_shard(&shard_file)?;
        shard.get_f32_tensor_owned(name)
    }
}

impl ShardedSafetensors {
    /// Load all Qwen3 weights from a sharded checkpoint into an owned backing store.
    ///
    /// Returns `(backing, weights)` where `weights` borrows from `backing`.
    /// The caller MUST keep `backing` alive until after `weights` is dropped.
    /// `QwenModel` enforces this with `ManuallyDrop` fields and an explicit
    /// `Drop` impl that drops `weights` before `_storage`.
    pub fn load_qwen_weights_owned(
        &mut self,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Result<(Box<ShardedQwenBacking>, QwenWeights<'static>), InferenceError> {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = q_dim + 2 * kv_dim;
        let gate_up_out_dim = 2 * intermediate_size;

        // --- Load global tensors ---
        let (embed_data, embed_shape) = self.get_f32_tensor_owned("embed_tokens.weight")?;
        if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "embed_tokens.weight".into(),
                expected: vec![embed_shape[0], hidden_size],
                actual: embed_shape,
            });
        }
        let embed_vocab = embed_shape[0];

        let (norm_data, norm_shape) = self.get_f32_tensor_owned("norm.weight")?;
        if norm_shape != [hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: "norm.weight".into(),
                expected: vec![hidden_size],
                actual: norm_shape,
            });
        }

        // --- Load per-layer tensors ---
        let mut q_proj_vecs = Vec::with_capacity(num_layers);
        let mut k_proj_vecs = Vec::with_capacity(num_layers);
        let mut v_proj_vecs = Vec::with_capacity(num_layers);
        let mut o_proj_vecs = Vec::with_capacity(num_layers);
        let mut q_norm_vecs = Vec::with_capacity(num_layers);
        let mut k_norm_vecs = Vec::with_capacity(num_layers);
        let mut input_ln_vecs = Vec::with_capacity(num_layers);
        let mut gate_proj_vecs = Vec::with_capacity(num_layers);
        let mut up_proj_vecs = Vec::with_capacity(num_layers);
        let mut down_proj_vecs = Vec::with_capacity(num_layers);
        let mut post_ln_vecs = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let p = format!("layers.{i}");

            let (q, qs) = self.get_f32_tensor_owned(&format!("{p}.self_attn.q_proj.weight"))?;
            if qs != [q_dim, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.q_proj.weight"),
                    expected: vec![q_dim, hidden_size],
                    actual: qs,
                });
            }
            q_proj_vecs.push(q);

            let (k, ks) = self.get_f32_tensor_owned(&format!("{p}.self_attn.k_proj.weight"))?;
            if ks != [kv_dim, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.k_proj.weight"),
                    expected: vec![kv_dim, hidden_size],
                    actual: ks,
                });
            }
            k_proj_vecs.push(k);

            let (v, vs) = self.get_f32_tensor_owned(&format!("{p}.self_attn.v_proj.weight"))?;
            if vs != [kv_dim, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.v_proj.weight"),
                    expected: vec![kv_dim, hidden_size],
                    actual: vs,
                });
            }
            v_proj_vecs.push(v);

            let (o, os) = self.get_f32_tensor_owned(&format!("{p}.self_attn.o_proj.weight"))?;
            if os != [hidden_size, q_dim] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.o_proj.weight"),
                    expected: vec![hidden_size, q_dim],
                    actual: os,
                });
            }
            o_proj_vecs.push(o);

            let (qn, qns) = self.get_f32_tensor_owned(&format!("{p}.self_attn.q_norm.weight"))?;
            if qns != [head_dim] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.q_norm.weight"),
                    expected: vec![head_dim],
                    actual: qns,
                });
            }
            q_norm_vecs.push(qn);

            let (kn, kns) = self.get_f32_tensor_owned(&format!("{p}.self_attn.k_norm.weight"))?;
            if kns != [head_dim] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.k_norm.weight"),
                    expected: vec![head_dim],
                    actual: kns,
                });
            }
            k_norm_vecs.push(kn);

            let (iln, ilns) = self.get_f32_tensor_owned(&format!("{p}.input_layernorm.weight"))?;
            if ilns != [hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.input_layernorm.weight"),
                    expected: vec![hidden_size],
                    actual: ilns,
                });
            }
            input_ln_vecs.push(iln);

            let (g, gs) = self.get_f32_tensor_owned(&format!("{p}.mlp.gate_proj.weight"))?;
            if gs != [intermediate_size, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.mlp.gate_proj.weight"),
                    expected: vec![intermediate_size, hidden_size],
                    actual: gs,
                });
            }
            gate_proj_vecs.push(g);

            let (u, us) = self.get_f32_tensor_owned(&format!("{p}.mlp.up_proj.weight"))?;
            if us != [intermediate_size, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.mlp.up_proj.weight"),
                    expected: vec![intermediate_size, hidden_size],
                    actual: us,
                });
            }
            up_proj_vecs.push(u);

            let (d, ds) = self.get_f32_tensor_owned(&format!("{p}.mlp.down_proj.weight"))?;
            if ds != [hidden_size, intermediate_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.mlp.down_proj.weight"),
                    expected: vec![hidden_size, intermediate_size],
                    actual: ds,
                });
            }
            down_proj_vecs.push(d);

            let (pln, plns) =
                self.get_f32_tensor_owned(&format!("{p}.post_attention_layernorm.weight"))?;
            if plns != [hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.post_attention_layernorm.weight"),
                    expected: vec![hidden_size],
                    actual: plns,
                });
            }
            post_ln_vecs.push(pln);
        }

        // --- Assemble owned backing ---
        let backing = Box::new(ShardedQwenBacking {
            embed_tokens: embed_data,
            norm_weight: norm_data,
            q_proj: q_proj_vecs,
            k_proj: k_proj_vecs,
            v_proj: v_proj_vecs,
            o_proj: o_proj_vecs,
            q_norm: q_norm_vecs,
            k_norm: k_norm_vecs,
            input_ln: input_ln_vecs,
            gate_proj: gate_proj_vecs,
            up_proj: up_proj_vecs,
            down_proj: down_proj_vecs,
            post_ln: post_ln_vecs,
        });

        // --- Build QwenWeights borrowing from the backing ---
        // SAFETY: The backing is heap-allocated in a Box; its address is stable.
        // `QwenModel` wraps both `weights` and `_storage` in `ManuallyDrop` and
        // implements `Drop` to drop `weights` before `_storage`, ensuring the
        // slice references in `weights` are released before `backing` is freed.
        // This invariant is independent of field declaration order in `QwenModel`.
        let weights: QwenWeights<'static> = {
            #[allow(clippy::explicit_auto_deref)]
            let b: &ShardedQwenBacking = &*backing;
            // SAFETY: We extend the borrow of `b` to 'static. This is sound because
            // `backing` outlives any use of these references: `QwenModel` drops
            // `weights` before `_storage` via an explicit `Drop` impl that uses
            // `ManuallyDrop`, independent of field declaration order.
            let b: &'static ShardedQwenBacking = unsafe { &*(b as *const ShardedQwenBacking) };

            let embed_tokens = Tensor2D {
                data: &b.embed_tokens,
                rows: embed_vocab,
                cols: hidden_size,
            };
            let norm_weight = Tensor1D {
                data: &b.norm_weight,
                len: hidden_size,
            };

            let mut layers = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                // Build fused QKV: [q_dim + 2*kv_dim, hidden_size]
                let mut fused_qkv = Vec::with_capacity(qkv_out_dim * hidden_size);
                fused_qkv.extend_from_slice(&b.q_proj[i]);
                fused_qkv.extend_from_slice(&b.k_proj[i]);
                fused_qkv.extend_from_slice(&b.v_proj[i]);

                // Build fused gate+up: [2*intermediate_size, hidden_size]
                let mut fused_gate_up = Vec::with_capacity(gate_up_out_dim * hidden_size);
                fused_gate_up.extend_from_slice(&b.gate_proj[i]);
                fused_gate_up.extend_from_slice(&b.up_proj[i]);

                layers.push(QwenLayerWeights {
                    q_proj_weight: Tensor2D {
                        data: &b.q_proj[i],
                        rows: q_dim,
                        cols: hidden_size,
                    },
                    k_proj_weight: Tensor2D {
                        data: &b.k_proj[i],
                        rows: kv_dim,
                        cols: hidden_size,
                    },
                    v_proj_weight: Tensor2D {
                        data: &b.v_proj[i],
                        rows: kv_dim,
                        cols: hidden_size,
                    },
                    o_proj_weight: Tensor2D {
                        data: &b.o_proj[i],
                        rows: hidden_size,
                        cols: q_dim,
                    },
                    q_norm_weight: Tensor1D {
                        data: &b.q_norm[i],
                        len: head_dim,
                    },
                    k_norm_weight: Tensor1D {
                        data: &b.k_norm[i],
                        len: head_dim,
                    },
                    input_layernorm_weight: Tensor1D {
                        data: &b.input_ln[i],
                        len: hidden_size,
                    },
                    gate_proj_weight: Tensor2D {
                        data: &b.gate_proj[i],
                        rows: intermediate_size,
                        cols: hidden_size,
                    },
                    up_proj_weight: Tensor2D {
                        data: &b.up_proj[i],
                        rows: intermediate_size,
                        cols: hidden_size,
                    },
                    down_proj_weight: Tensor2D {
                        data: &b.down_proj[i],
                        rows: hidden_size,
                        cols: intermediate_size,
                    },
                    post_attention_layernorm_weight: Tensor1D {
                        data: &b.post_ln[i],
                        len: hidden_size,
                    },
                    fused_qkv,
                    qkv_out_dim,
                    fused_gate_up,
                    gate_up_out_dim,
                });
            }

            QwenWeights {
                embed_tokens,
                norm_weight,
                layers,
            }
        };

        Ok((backing, weights))
    }
}

fn parse_safetensors_header(json: &str) -> Result<HashMap<String, TensorMeta>, InferenceError> {
    let mut parser = JsonParser::new(json);
    parser.expect(b'{')?;

    let mut tensors = HashMap::new();
    let mut seen_keys = HashSet::new();

    loop {
        parser.skip_ws();
        match parser.peek() {
            Some(b'}') => {
                parser.bump();
                break;
            }
            Some(_) => {}
            None => {
                return Err(InferenceError::InvalidSafetensors(
                    "unexpected end of header while parsing top-level object".into(),
                ));
            }
        }

        let key = parser.parse_string()?;
        if !seen_keys.insert(key.clone()) {
            let kind = if key == "__metadata__" {
                "metadata member"
            } else {
                "tensor name"
            };
            return Err(InferenceError::InvalidSafetensors(format!(
                "duplicate {kind} in safetensors header: {key}"
            )));
        }
        parser.skip_ws();
        parser.expect(b':')?;
        parser.skip_ws();

        if key == "__metadata__" {
            parser.parse_metadata()?;
        } else {
            let meta = parser.parse_tensor_meta(&key)?;
            tensors.insert(key, meta);
        }

        parser.skip_ws();
        match parser.peek() {
            Some(b',') => {
                parser.bump();
                parser.skip_ws();
                if matches!(parser.peek(), Some(b'}')) {
                    return Err(InferenceError::InvalidSafetensors(
                        "trailing comma in safetensors header object".into(),
                    ));
                }
            }
            Some(b'}') => {
                parser.bump();
                break;
            }
            Some(other) => {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "expected ',' or '}}' in header, found byte {other}"
                )));
            }
            None => {
                return Err(InferenceError::InvalidSafetensors(
                    "unexpected end of header after tensor entry".into(),
                ));
            }
        }
    }

    while matches!(parser.peek(), Some(b' ')) {
        parser.bump();
    }
    if let Some(other) = parser.peek() {
        return Err(InferenceError::InvalidSafetensors(format!(
            "non-space byte {other} after top-level safetensors header object"
        )));
    }

    Ok(tensors)
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// Current container nesting depth, checked against [`MAX_SAFETENSORS_HEADER_DEPTH`]
    /// on entry to each `{`/`[` container in [`Self::skip_value`].
    depth: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
            depth: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), InferenceError> {
        match self.bump() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(InferenceError::InvalidSafetensors(format!(
                "expected byte {:?}, found byte {:?} at position {}",
                expected as char,
                actual as char,
                self.pos.saturating_sub(1)
            ))),
            None => Err(InferenceError::InvalidSafetensors(format!(
                "expected byte {:?}, found end of input",
                expected as char
            ))),
        }
    }

    fn parse_string(&mut self) -> Result<String, InferenceError> {
        let start = self.pos;
        self.expect(b'"')?;
        let mut escaped = false;
        loop {
            let byte = self.bump().ok_or_else(|| {
                InferenceError::InvalidSafetensors("unterminated string in header".into())
            })?;
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                break;
            }
        }
        let raw = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|err| {
            InferenceError::InvalidSafetensors(format!(
                "safetensors header string is not valid UTF-8: {err}"
            ))
        })?;
        serde_json::from_str(raw).map_err(|err| {
            InferenceError::InvalidSafetensors(format!(
                "invalid JSON string in safetensors header: {err}"
            ))
        })
    }

    fn parse_usize(&mut self) -> Result<usize, InferenceError> {
        let start = self.pos;
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "leading zero in unsigned integer at byte {start}"
                    )));
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => {}
        }
        if start == self.pos {
            return Err(InferenceError::InvalidSafetensors(format!(
                "expected unsigned integer at byte {}",
                self.pos
            )));
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| {
            InferenceError::InvalidSafetensors(format!("invalid number token in header: {e}"))
        })?;
        s.parse::<usize>().map_err(|e| {
            InferenceError::InvalidSafetensors(format!("invalid usize value {s}: {e}"))
        })
    }

    fn parse_usize_array(&mut self) -> Result<Vec<usize>, InferenceError> {
        self.expect(b'[')?;
        self.skip_ws();
        let mut values = Vec::new();
        if matches!(self.peek(), Some(b']')) {
            self.bump();
            return Ok(values);
        }
        loop {
            self.skip_ws();
            values.push(self.parse_usize()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                    self.skip_ws();
                }
                Some(b']') => {
                    self.bump();
                    break;
                }
                Some(other) => {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "expected ',' or ']' in array, found byte {other}"
                    )));
                }
                None => {
                    return Err(InferenceError::InvalidSafetensors(
                        "unexpected end of input in array".into(),
                    ));
                }
            }
        }
        Ok(values)
    }

    /// Parse the optional safetensors `__metadata__` member. The wire format
    /// requires a JSON object whose keys and values are strings; arbitrary
    /// nested JSON is not a valid metadata payload.
    fn parse_metadata(&mut self) -> Result<(), InferenceError> {
        self.expect(b'{')?;
        self.skip_ws();
        let mut seen_keys = HashSet::new();
        if matches!(self.peek(), Some(b'}')) {
            self.bump();
            return Ok(());
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            if !seen_keys.insert(key.clone()) {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "duplicate __metadata__ key in safetensors header: {key}"
                )));
            }
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            self.parse_string().map_err(|_| {
                InferenceError::InvalidSafetensors(format!(
                    "safetensors __metadata__ value for {key:?} must be a string"
                ))
            })?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                }
                Some(b'}') => {
                    self.bump();
                    break;
                }
                Some(other) => {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "expected ',' or '}}' in safetensors __metadata__, found byte {other}"
                    )));
                }
                None => {
                    return Err(InferenceError::InvalidSafetensors(
                        "unexpected end of safetensors __metadata__ object".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn skip_value(&mut self) -> Result<(), InferenceError> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => {
                self.parse_string()?;
                Ok(())
            }
            Some(b'{') => {
                self.depth += 1;
                if self.depth > MAX_SAFETENSORS_HEADER_DEPTH {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "header nesting exceeds depth limit of {MAX_SAFETENSORS_HEADER_DEPTH}"
                    )));
                }
                self.bump();
                self.skip_ws();
                if matches!(self.peek(), Some(b'}')) {
                    self.bump();
                    self.depth -= 1;
                    return Ok(());
                }
                let mut seen_keys = HashSet::new();
                loop {
                    self.skip_ws();
                    let key = self.parse_string()?;
                    if !seen_keys.insert(key.clone()) {
                        return Err(InferenceError::InvalidSafetensors(format!(
                            "duplicate key in safetensors header object: {key}"
                        )));
                    }
                    self.skip_ws();
                    self.expect(b':')?;
                    self.skip_ws();
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.bump();
                        }
                        Some(b'}') => {
                            self.bump();
                            break;
                        }
                        Some(other) => {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "expected ',' or '}}' while skipping object, found {other}"
                            )));
                        }
                        None => {
                            return Err(InferenceError::InvalidSafetensors(
                                "unexpected end of input while skipping object".into(),
                            ));
                        }
                    }
                }
                self.depth -= 1;
                Ok(())
            }
            Some(b'[') => {
                self.depth += 1;
                if self.depth > MAX_SAFETENSORS_HEADER_DEPTH {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "header nesting exceeds depth limit of {MAX_SAFETENSORS_HEADER_DEPTH}"
                    )));
                }
                self.bump();
                self.skip_ws();
                if matches!(self.peek(), Some(b']')) {
                    self.bump();
                    self.depth -= 1;
                    return Ok(());
                }
                loop {
                    self.skip_ws();
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.bump();
                        }
                        Some(b']') => {
                            self.bump();
                            break;
                        }
                        Some(other) => {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "expected ',' or ']' while skipping array, found {other}"
                            )));
                        }
                        None => {
                            return Err(InferenceError::InvalidSafetensors(
                                "unexpected end of input while skipping array".into(),
                            ));
                        }
                    }
                }
                self.depth -= 1;
                Ok(())
            }
            Some(b't') => self.skip_literal(b"true"),
            Some(b'f') => self.skip_literal(b"false"),
            Some(b'n') => self.skip_literal(b"null"),
            Some(b'-' | b'0'..=b'9') => self.skip_number(),
            Some(other) => Err(InferenceError::InvalidSafetensors(format!(
                "unsupported JSON value starting with byte {other}"
            ))),
            None => Err(InferenceError::InvalidSafetensors(
                "unexpected end of input while skipping value".into(),
            )),
        }
    }

    fn skip_literal(&mut self, literal: &[u8]) -> Result<(), InferenceError> {
        if self.pos + literal.len() > self.bytes.len()
            || &self.bytes[self.pos..self.pos + literal.len()] != literal
        {
            return Err(InferenceError::InvalidSafetensors(format!(
                "expected literal {:?}",
                std::str::from_utf8(literal).unwrap_or("<literal>")
            )));
        }
        self.pos += literal.len();
        Ok(())
    }

    /// Consume one RFC 8259 JSON number. This is used only for unknown
    /// extension values, but accepting a loose run of number-like bytes here
    /// would make the entire safetensors header non-JSON while still passing
    /// structural preflight.
    fn skip_number(&mut self) -> Result<(), InferenceError> {
        let start = self.pos;
        if matches!(self.peek(), Some(b'-')) {
            self.pos += 1;
        }

        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "leading zero in JSON number at byte {start}"
                    )));
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "invalid JSON number at byte {start}: expected an integer digit"
                )));
            }
        }

        if matches!(self.peek(), Some(b'.')) {
            self.pos += 1;
            let fraction_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == fraction_start {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "invalid JSON number at byte {start}: fraction has no digits"
                )));
            }
        }

        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exponent_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == exponent_start {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "invalid JSON number at byte {start}: exponent has no digits"
                )));
            }
        }

        Ok(())
    }

    /// Parse one tensor's header entry. `tensor_name` is the key this entry
    /// was found under, threaded through purely for error messages.
    ///
    /// Every entry with a shape and data_offsets is preserved — including
    /// dtypes this crate doesn't materialize as f32 — so extent validation
    /// covers them and a later *request* for such a tensor fails with a
    /// clear dtype/format error instead of looking like a missing tensor
    /// (lattice#800). Only a dtype string this crate has never heard of is
    /// rejected here, at parse time, since its byte width is unknowable and
    /// extent validation cannot proceed without it.
    fn parse_tensor_meta(&mut self, tensor_name: &str) -> Result<TensorMeta, InferenceError> {
        self.expect(b'{')?;
        self.skip_ws();
        let mut dtype: Option<DType> = None;
        let mut dtype_str: Option<String> = None;
        let mut shape = None;
        let mut data_offsets = None;
        let mut seen_keys = HashSet::new();

        if matches!(self.peek(), Some(b'}')) {
            self.bump();
        } else {
            loop {
                self.skip_ws();
                let key = self.parse_string()?;
                if !seen_keys.insert(key.clone()) {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "duplicate member in tensor {tensor_name} header object: {key}"
                    )));
                }
                self.skip_ws();
                self.expect(b':')?;
                self.skip_ws();

                match key.as_str() {
                    "dtype" => {
                        let s = self.parse_string()?;
                        dtype = dtype_from_str(&s);
                        dtype_str = Some(s);
                    }
                    "shape" => shape = Some(self.parse_usize_array()?),
                    "data_offsets" => {
                        let arr = self.parse_usize_array()?;
                        if arr.len() != 2 {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "data_offsets must have length 2, got {}",
                                arr.len()
                            )));
                        }
                        data_offsets = Some((arr[0], arr[1]));
                    }
                    _ => self.skip_value()?,
                }

                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.bump();
                    }
                    Some(b'}') => {
                        self.bump();
                        break;
                    }
                    Some(other) => {
                        return Err(InferenceError::InvalidSafetensors(format!(
                            "expected ',' or '}}' in tensor object, found {other}"
                        )));
                    }
                    None => {
                        return Err(InferenceError::InvalidSafetensors(
                            "unexpected end of input in tensor object".into(),
                        ));
                    }
                }
            }
        }

        let dtype = match (dtype, dtype_str) {
            (Some(dtype), _) => dtype,
            (None, Some(s)) => {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {tensor_name} has unrecognized dtype {s:?}; its byte width is \
                     unknowable so extent validation cannot proceed"
                )));
            }
            (None, None) => {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {tensor_name} entry missing dtype"
                )));
            }
        };
        let shape = shape.ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!("tensor {tensor_name} entry missing shape"))
        })?;
        let (start, end) = data_offsets.ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "tensor {tensor_name} entry missing data_offsets"
            ))
        })?;

        Ok(TensorMeta {
            dtype,
            shape,
            start,
            end,
            converted_f32: OnceLock::new(),
            validated: OnceLock::new(),
        })
    }
}

#[cfg(test)]
mod tests;
