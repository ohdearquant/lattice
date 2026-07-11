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
//! Scope note: this issue wires up the "decoded F32" payload form only —
//! safetensors F32/F16/BF16 all normalize to `f32` by the time they reach
//! this seam (`SafetensorsFile::get_f32_tensor` widens F16/BF16 before
//! returning). The other payload forms named in lattice#800 — raw
//! safetensors F32/F16/BF16 bytes, decoded F64, native Q4 blocks/bytes, and
//! Q8 data+scales — are added by the companion issues in this cluster
//! (QuaRot/offline-quantizer routing, native Q4/KHF1 validation
//! unification, Q8 routing) as those loaders start calling into this seam.

use crate::error::InferenceError;

/// Sealed carrier for one tensor's ingested payload plus its provenance.
///
/// Construction only happens through [`IngestedTensor::decoded_f32`], so a
/// call site cannot fabricate an already-"validated" value and skip
/// [`validate_ingested_tensor`].
#[derive(Debug)]
pub(crate) struct IngestedTensor<'a> {
    source: &'a str,
    tensor_name: &'a str,
    shape: &'a [usize],
    payload: IngestPayload<'a>,
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
}

/// Validate one ingested tensor at the exact point its bytes become trusted
/// tensor data: the declared shape's element count must not overflow, the
/// payload's element count must exactly match that shape, and every element
/// must be finite (NaN and either infinity are rejected; signed zero and
/// subnormals are accepted).
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
