//! Private weight-ingress validation seam (lattice#800).
//!
//! Weight-loading correctness checks (finite-value scans, shape/extent
//! validation) used to be scattered per-loader with materially different
//! coverage — `SafetensorsFile::get_f32_tensor` validated header bounds and
//! byte length at open time, but silently accepted NaN/Inf in the decoded
//! tensor data itself (lattice#793). This module is the one seam every
//! external-bytes-to-tensor ingestion path in this crate is expected to
//! route through immediately before a tensor view escapes to a caller, so
//! the same class of gap cannot silently reopen at the next loader.
//!
//! Safetensors runtime loads arrive as decoded `f32` slices. QuaRot reads
//! arrive as raw F32/F16/BF16 bytes and are decoded directly into their
//! one-tensor `Vec<f64>` here, so finiteness is checked during the existing
//! materialization pass. KHF1 output uses the same seam while narrowing to
//! f16, which prevents a finite source value that overflows f16 from being
//! published as non-finite output.

use crate::error::InferenceError;

/// Sealed carrier for one tensor's ingested payload plus its provenance.
///
/// Construction happens through [`IngestedTensor::decoded_f32`] (already
/// f32-widened data), [`IngestedTensor::decode_f64`] (raw on-disk bytes
/// decoded to f64), or [`IngestedTensor::encode_f16`] (f64 encoded down to
/// f16 output bytes) — a call site cannot fabricate an already-"validated"
/// value and skip [`validate_ingested_tensor`].
#[derive(Debug)]
pub(crate) struct IngestedTensor<'a> {
    source: &'a str,
    tensor_name: &'a str,
    shape: &'a [usize],
    payload: IngestPayload<'a>,
}

/// On-disk raw float dtype for [`IngestedTensor::decode_f64`].
///
/// Matched exactly once per call in [`validate_ingested_tensor`], which then
/// runs a per-dtype tight decode loop with the conversion inlined at the
/// call site. The previous design stored a `fn(&[u8]) -> f64` pointer
/// selected by a match and invoked it per element; release LLVM IR showed
/// that lowering to a per-element indirect `tail call`, defeating inlining
/// for a hot per-tensor loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawDType {
    F32,
    F16,
    Bf16,
}

#[derive(Debug)]
enum IngestPayload<'a> {
    /// Already widened to `f32`. `dtype_label` names the *source* dtype
    /// ("F32", "F16", "BF16") for error messages — the slice itself is
    /// always `f32` by the time it reaches this seam.
    DecodedF32 {
        values: &'a [f32],
        dtype_label: &'static str,
    },
    DecodeF64 {
        bytes: &'a [u8],
        dtype: RawDType,
        values: &'a mut Vec<f64>,
        dtype_label: &'static str,
    },
    EncodeF16 {
        values: &'a [f64],
        bytes: &'a mut Vec<u8>,
    },
}

impl<'a> IngestedTensor<'a> {
    /// Wrap an already f32-widened tensor slice for validation.
    pub(crate) fn decoded_f32(
        source: &'a str,
        tensor_name: &'a str,
        shape: &'a [usize],
        dtype_label: &'static str,
        values: &'a [f32],
    ) -> Self {
        Self {
            source,
            tensor_name,
            shape,
            payload: IngestPayload::DecodedF32 {
                values,
                dtype_label,
            },
        }
    }

    /// Wrap raw floating-point bytes for checked one-pass f64 decoding.
    pub(crate) fn decode_f64(
        source: &'a str,
        tensor_name: &'a str,
        shape: &'a [usize],
        dtype_label: &'static str,
        bytes: &'a [u8],
        dtype: RawDType,
        values: &'a mut Vec<f64>,
    ) -> Self {
        Self {
            source,
            tensor_name,
            shape,
            payload: IngestPayload::DecodeF64 {
                bytes,
                dtype,
                values,
                dtype_label,
            },
        }
    }

    /// Wrap f64 values for checked one-pass f16 output encoding.
    pub(crate) fn encode_f16(
        source: &'a str,
        tensor_name: &'a str,
        shape: &'a [usize],
        values: &'a [f64],
        bytes: &'a mut Vec<u8>,
    ) -> Self {
        Self {
            source,
            tensor_name,
            shape,
            payload: IngestPayload::EncodeF16 { values, bytes },
        }
    }
}

/// Validate one ingested tensor at the exact point its bytes become trusted
/// tensor data: the declared shape's element count must not overflow, the
/// payload's element count must exactly match that shape, and every trusted
/// element must be finite (NaN and either infinity are rejected; signed zero
/// and subnormals are accepted). Raw-to-f64 decode and f64-to-f16 encode are
/// validated while their output vectors are built, without a second scan.
///
/// Errors are `InferenceError::InvalidSafetensors` — no new public error
/// variant is added — and carry source label, tensor name, dtype, shape,
/// and (for a finite-value failure) the first offending element index.
pub(crate) fn validate_ingested_tensor(tensor: IngestedTensor<'_>) -> Result<(), InferenceError> {
    let numel = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "{}: tensor {} shape {:?} overflows usize element count",
                tensor.source, tensor.tensor_name, tensor.shape,
            ))
        })?;

    match tensor.payload {
        IngestPayload::DecodedF32 {
            values,
            dtype_label,
        } => {
            if values.len() != numel {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} ({dtype_label}) decoded element count {} does not match \
                     shape {:?} (expected {numel})",
                    tensor.source,
                    tensor.tensor_name,
                    values.len(),
                    tensor.shape,
                )));
            }
            if let Some((idx, bad)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} ({dtype_label}) has non-finite value {bad} at element index \
                     {idx} of {numel} (shape {:?})",
                    tensor.source, tensor.tensor_name, tensor.shape,
                )));
            }
            Ok(())
        }
        IngestPayload::DecodeF64 {
            bytes,
            dtype,
            values,
            dtype_label,
        } => {
            let bytes_per_element = match dtype {
                RawDType::F32 => 4,
                RawDType::F16 | RawDType::Bf16 => 2,
            };
            let expected_bytes = numel.checked_mul(bytes_per_element).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} ({dtype_label}) byte length overflows usize for shape {:?}",
                    tensor.source, tensor.tensor_name, tensor.shape,
                ))
            })?;
            if bytes.len() != expected_bytes {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} ({dtype_label}) byte length {} does not match shape {:?} \
                     (expected {expected_bytes})",
                    tensor.source,
                    tensor.tensor_name,
                    bytes.len(),
                    tensor.shape,
                )));
            }

            values.clear();
            values.reserve_exact(numel);

            // `dtype` is matched exactly once here; each arm is a tight loop
            // with its decode statically known at the call site (no stored
            // fn-pointer indirection — see `RawDType` doc comment).
            macro_rules! decode_loop {
                ($chunk_len:expr, $decode:expr) => {
                    for (idx, chunk) in bytes.chunks_exact($chunk_len).enumerate() {
                        let value: f64 = $decode(chunk);
                        if !value.is_finite() {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "{}: tensor {} ({dtype_label}) has non-finite value {value} at \
                                 element index {idx} of {numel} (shape {:?})",
                                tensor.source, tensor.tensor_name, tensor.shape,
                            )));
                        }
                        values.push(value);
                    }
                };
            }

            match dtype {
                RawDType::F32 => decode_loop!(4, |chunk: &[u8]| f32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3]
                ]) as f64),
                RawDType::F16 => {
                    decode_loop!(
                        2,
                        |chunk: &[u8]| crate::weights::half_bits::f16_bits_to_f32(
                            u16::from_le_bytes([chunk[0], chunk[1]])
                        ) as f64
                    )
                }
                RawDType::Bf16 => {
                    decode_loop!(
                        2,
                        |chunk: &[u8]| crate::weights::half_bits::bf16_bits_to_f32(
                            u16::from_le_bytes([chunk[0], chunk[1]])
                        ) as f64
                    )
                }
            }
            Ok(())
        }
        IngestPayload::EncodeF16 { values, bytes } => {
            if values.len() != numel {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} (F16) source element count {} does not match shape {:?} \
                     (expected {numel})",
                    tensor.source,
                    tensor.tensor_name,
                    values.len(),
                    tensor.shape,
                )));
            }

            let encoded_len = numel.checked_mul(2).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} (F16) encoded byte length overflows usize for shape {:?}",
                    tensor.source, tensor.tensor_name, tensor.shape,
                ))
            })?;
            bytes.clear();
            bytes.reserve_exact(encoded_len);
            for (idx, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "{}: tensor {} (F16) has non-finite source value {value} at element \
                         index {idx} of {numel} (shape {:?})",
                        tensor.source, tensor.tensor_name, tensor.shape,
                    )));
                }
                let narrowed = value as f32;
                let bits = crate::weights::half_bits::f32_to_f16_bits(narrowed);
                if !crate::weights::half_bits::f16_bits_is_finite(bits) {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "{}: tensor {} (F16) encodes finite source value {value} as a \
                         non-finite f16 bit pattern at element index {idx} of {numel} \
                         (shape {:?})",
                        tensor.source, tensor.tensor_name, tensor.shape,
                    )));
                }
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_finite_values_including_signed_zero_and_subnormal() {
        let values = [0.0f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE / 2.0];
        let tensor = IngestedTensor::decoded_f32("test", "t", &[5], "F32", &values);
        assert!(validate_ingested_tensor(tensor).is_ok());
    }

    #[test]
    fn rejects_nan() {
        let values = [1.0f32, f32::NAN, 3.0];
        let tensor = IngestedTensor::decoded_f32("test", "t", &[3], "F32", &values);
        let err = validate_ingested_tensor(tensor).expect_err("NaN must be rejected");
        let msg = err.to_string();
        assert!(msg.contains('t'), "error should name the tensor: {msg}");
        assert!(
            msg.contains("element index 1"),
            "error should point at the offending index: {msg}"
        );
    }

    #[test]
    fn rejects_positive_infinity() {
        let values = [1.0f32, f32::INFINITY];
        let tensor = IngestedTensor::decoded_f32("test", "t", &[2], "F16", &values);
        let err = validate_ingested_tensor(tensor).expect_err("+inf must be rejected");
        assert!(err.to_string().contains("element index 1"));
    }

    #[test]
    fn rejects_negative_infinity() {
        let values = [f32::NEG_INFINITY, 2.0];
        let tensor = IngestedTensor::decoded_f32("test", "t", &[2], "BF16", &values);
        let err = validate_ingested_tensor(tensor).expect_err("-inf must be rejected");
        assert!(err.to_string().contains("element index 0"));
    }

    #[test]
    fn rejects_shape_product_overflow() {
        let values: [f32; 0] = [];
        let huge_shape = [usize::MAX, 2];
        let tensor = IngestedTensor::decoded_f32("test", "t", &huge_shape, "F32", &values);
        let err = validate_ingested_tensor(tensor).expect_err("overflow must be rejected");
        assert!(err.to_string().contains("overflows"));
    }

    #[test]
    fn rejects_element_count_mismatch() {
        let values = [1.0f32, 2.0, 3.0];
        // Shape says 4 elements, payload only has 3.
        let tensor = IngestedTensor::decoded_f32("test", "t", &[4], "F32", &values);
        let err = validate_ingested_tensor(tensor).expect_err("count mismatch must be rejected");
        assert!(err.to_string().contains("does not match"));
    }

    /// Mutation sensitivity: with the `!v.is_finite()` scan removed (or
    /// replaced by an always-`Ok` stub), this test's NaN input would pass
    /// validation. Verified by temporarily deleting the scan and confirming
    /// `rejects_nan` above goes red before restoring it (see PR description
    /// for the revert/touch/restore cycle).
    #[test]
    fn all_of_a_finite_tensor_is_scanned_not_just_the_first_element() {
        let mut values = vec![1.0f32; 64];
        values[63] = f32::NAN;
        let tensor = IngestedTensor::decoded_f32("test", "t", &[64], "F32", &values);
        let err = validate_ingested_tensor(tensor).expect_err("tail NaN must be caught");
        assert!(err.to_string().contains("element index 63"));
    }
}
