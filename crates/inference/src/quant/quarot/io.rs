//! Streaming SafeTensors reader for the QuaRot conversion pipeline.
//!
//! The QuaRot offline converter (step 3c) needs to read every weight tensor
//! exactly once, promote it to f64 for the rotation math, fuse RMSNorm scales,
//! apply rotation absorption, then quantize and discard. Caching the f32 or
//! f64 expansion of a multi-gigabyte checkpoint would defeat that pipeline —
//! so this reader allocates a fresh `Vec<f64>` per [`read_tensor_f64`] call
//! and never caches converted bytes. Compare with
//! [`crate::weights::f32_weights::SafetensorsFile`] which DOES cache the
//! f32-converted form per tensor and is appropriate for inference paths
//! that revisit weights repeatedly.
//!
//! Layout is auto-detected:
//!
//! - `model.safetensors.index.json` present → sharded layout. Shards are
//!   opened lazily on first access to a tensor they contain and re-used
//!   for subsequent reads, mirroring
//!   [`crate::weights::f32_weights::ShardedSafetensors`].
//! - `model.safetensors` present → single-file layout.
//! - Otherwise: error at [`QuarotTensorReader::open`].
//!
//! On-disk decode for F32 / F16 / BF16 is hand-rolled to keep this module
//! independent of the `f16` cargo feature. The conversion is bit-identical
//! to [`crate::weights::f32_weights`]'s internal `f16_to_f32` /
//! `bf16_to_f32`, then widened to f64 (lossless from f32).
//!
//! Step 3b deliverable per [ADR-044]; consumed by step 3c's
//! `quantize_quarot` binary.
//!
//! [ADR-044]: ../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde_json::Value;

use crate::error::InferenceError;
use crate::weights::f32_weights::parse_index;

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

    fn bytes_per_elem(self) -> usize {
        match self {
            SourceDType::F32 => 4,
            SourceDType::F16 | SourceDType::BF16 => 2,
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
    headers: HashMap<String, TensorHeader>,
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

        let mut headers = HashMap::with_capacity(obj.len());
        let data_len = mmap.len() - data_offset;

        for (name, entry) in obj {
            if name == "__metadata__" {
                continue;
            }

            let dtype_str = entry.get("dtype").and_then(Value::as_str).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: missing or non-string dtype"
                ))
            })?;
            let Some(dtype) = SourceDType::from_header_str(dtype_str) else {
                // Tensors with unsupported dtypes (I32, I64, BOOL, …) are
                // skipped silently. The converter only requests weight
                // tensors (always floating-point) so this is safe.
                continue;
            };

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
            let expected = numel.checked_mul(dtype.bytes_per_elem()).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: byte length overflows usize"
                ))
            })?;
            let actual = end - start;
            if actual != expected {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name}: byte length mismatch for {} {:?}: \
                     expected {expected}, got {actual}",
                    dtype.name(),
                    shape
                )));
            }

            headers.insert(
                name.clone(),
                TensorHeader {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
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
        root: PathBuf,
        /// Tensor name → shard file name (relative to `root`).
        weight_map: HashMap<String, String>,
        /// Lazily opened shards, keyed by shard file name.
        shards: HashMap<String, Shard>,
    },
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
    /// Detection order:
    /// 1. `model.safetensors.index.json` present → sharded
    /// 2. `model.safetensors` present → single file
    /// 3. neither → [`InferenceError::InvalidSafetensors`]
    ///
    /// When both are present (some HuggingFace repos include both) the
    /// sharded index wins, matching what the runtime loader does.
    pub fn open(model_dir: &Path) -> Result<Self, InferenceError> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let single_path = model_dir.join("model.safetensors");

        if index_path.exists() {
            let index = parse_index(model_dir)?;
            Ok(Self {
                backing: Backing::Sharded {
                    root: model_dir.to_path_buf(),
                    weight_map: index.weight_map,
                    shards: HashMap::new(),
                },
            })
        } else if single_path.exists() {
            let shard = Shard::open(&single_path)?;
            Ok(Self {
                backing: Backing::Single { shard },
            })
        } else {
            Err(InferenceError::InvalidSafetensors(format!(
                "{}: missing both model.safetensors and \
                 model.safetensors.index.json",
                model_dir.display()
            )))
        }
    }

    /// All known tensor names.
    ///
    /// In sharded mode this consults the index file's weight map without
    /// opening any shards, so it is cheap to call.
    pub fn tensor_names(&self) -> Vec<String> {
        match &self.backing {
            Backing::Single { shard } => shard.headers.keys().cloned().collect(),
            Backing::Sharded { weight_map, .. } => weight_map.keys().cloned().collect(),
        }
    }

    /// Whether the model declares a tensor with the given name.
    ///
    /// In sharded mode this consults the index file's weight map without
    /// opening the containing shard.
    pub fn has_tensor(&self, name: &str) -> bool {
        match &self.backing {
            Backing::Single { shard } => shard.headers.contains_key(name),
            Backing::Sharded { weight_map, .. } => weight_map.contains_key(name),
        }
    }

    /// Shape of a named tensor.
    ///
    /// In sharded mode this may open the containing shard on first access.
    pub fn tensor_shape(&mut self, name: &str) -> Result<Vec<usize>, InferenceError> {
        let shard = self.ensure_shard_for(name)?;
        shard
            .headers
            .get(name)
            .map(|h| h.shape.clone())
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    /// Source dtype of a named tensor as stored on disk.
    ///
    /// In sharded mode this may open the containing shard on first access.
    pub fn source_dtype(&mut self, name: &str) -> Result<SourceDType, InferenceError> {
        let shard = self.ensure_shard_for(name)?;
        shard
            .headers
            .get(name)
            .map(|h| h.dtype)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    /// Read a tensor and convert to a fresh `Vec<f64>`, returned alongside
    /// the shape. Element order is row-major (the on-disk safetensors
    /// convention).
    ///
    /// No conversion cache is kept — the converter discards tensors after
    /// rotation/fusion so caching would only bloat memory.
    pub fn read_tensor_f64(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f64>, Vec<usize>), InferenceError> {
        let shard = self.ensure_shard_for(name)?;
        let header = shard
            .headers
            .get(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?
            .clone();
        let bytes = shard.tensor_bytes(name)?;
        let data = decode_bytes_to_f64(bytes, header.dtype)?;
        Ok((data, header.shape))
    }

    fn ensure_shard_for(&mut self, name: &str) -> Result<&Shard, InferenceError> {
        match &mut self.backing {
            Backing::Single { shard } => {
                if !shard.headers.contains_key(name) {
                    return Err(InferenceError::MissingTensor(name.to_string()));
                }
                Ok(shard)
            }
            Backing::Sharded {
                root,
                weight_map,
                shards,
            } => {
                let shard_file = weight_map
                    .get(name)
                    .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?
                    .clone();
                if !shards.contains_key(&shard_file) {
                    let path = root.join(&shard_file);
                    let new_shard = Shard::open(&path)?;
                    shards.insert(shard_file.clone(), new_shard);
                }
                shards.get(&shard_file).ok_or_else(|| {
                    InferenceError::InvalidSafetensors(format!(
                        "internal: failed to cache shard {shard_file}"
                    ))
                })
            }
        }
    }
}

fn decode_bytes_to_f64(bytes: &[u8], dtype: SourceDType) -> Result<Vec<f64>, InferenceError> {
    match dtype {
        SourceDType::F32 => {
            if bytes.len() % 4 != 0 {
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
            if bytes.len() % 2 != 0 {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "BF16 tensor byte length {} not divisible by 2",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| {
                    let bf16 = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bf16 as u32) << 16) as f64
                })
                .collect())
        }
        SourceDType::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "F16 tensor byte length {} not divisible by 2",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64)
                .collect())
        }
    }
}

/// IEEE-754 binary16 → f32. Bit-identical to the internal helper in
/// [`crate::weights::f32_weights`] (which is `#[cfg(feature = "f16")]`-gated
/// and so unavailable here without dragging the feature flag into QuaRot).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            // Subnormal: shift the leading 1 into bit 10, then strip it
            // and treat the rest as f32 mantissa.
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };

    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let mut reader = QuarotTensorReader::open(dir.path()).unwrap();
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

        let mut reader = QuarotTensorReader::open(dir.path()).unwrap();
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

        let mut reader = QuarotTensorReader::open(dir.path()).unwrap();
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

        let mut reader = QuarotTensorReader::open(dir.path()).unwrap();
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
        let mut reader = QuarotTensorReader::open(dir.path()).unwrap();
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

        let mut reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("supported"));
        assert!(!reader.has_tensor("ignored"));
        let (data, _) = reader.read_tensor_f64("supported").unwrap();
        assert_eq!(data, vec![7.0]);
    }

    #[test]
    fn sharded_index_wins_when_both_present() {
        // If both single-file and sharded index are present, the index path
        // is honored. Build a sharded layout where the "single" file is
        // a decoy with wrong content.
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("decoy", FixtureDType::F32, vec![1], &[999.0])],
        );
        write_safetensors(
            &dir.path().join("model-00001-of-00001.safetensors"),
            &[("real", FixtureDType::F32, vec![1], &[1.0])],
        );
        let index = serde_json::json!({
            "metadata": {"total_size": 4usize},
            "weight_map": {"real": "model-00001-of-00001.safetensors"},
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let mut reader = QuarotTensorReader::open(dir.path()).unwrap();
        assert!(reader.has_tensor("real"));
        assert!(!reader.has_tensor("decoy"));
        let (data, _) = reader.read_tensor_f64("real").unwrap();
        assert_eq!(data, vec![1.0]);
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
