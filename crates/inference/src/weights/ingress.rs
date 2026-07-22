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
//! Safetensors F32/F16/BF16 and native KHF1 normalize to `f32` before this
//! seam (`SafetensorsFile::get_f32_tensor` widens F16/BF16 before
//! returning); native Q4 files stream their block metadata through it
//! without materializing an additional payload copy. In-memory Q8
//! construction validates its decoded f32 source before quantization and
//! its derived Q8 data/scales before returning, so the seam covers the
//! conversion boundary as a whole. Other payload forms named in
//! lattice#800 are added as their loaders start calling into this seam.

use crate::error::InferenceError;

/// Sealed carrier for one tensor's ingested payload plus its provenance.
///
/// Construction only happens through the payload-specific constructors, so
/// a call site cannot fabricate an already-"validated" value and skip
/// [`validate_ingested_tensor`].
#[derive(Debug)]
pub(crate) struct IngestedTensor<'a> {
    source: &'a str,
    tensor_name: &'a str,
    shape: &'a [usize],
    expected_shape: Option<&'a [usize]>,
    payload: IngestPayload<'a>,
}

#[derive(Debug, Clone, Copy)]
enum IngestPayload<'a> {
    /// Already widened to `f32`. `dtype_label` names the *source* dtype
    /// ("F32", "F16", "BF16") for error messages — the slice itself is
    /// always `f32` by the time it reaches this seam.
    DecodedF32 {
        values: &'a [f32],
        dtype_label: &'static str,
        error_kind: IngressErrorKind,
    },
    /// Derived Q8 data and per-row scales.
    Q8 { data: &'a [i8], scales: &'a [f32] },
    NativeQ4 {
        original_len: usize,
        block_count: usize,
    },
    NativeQ4Block(Q4BlockMetadata),
}

#[derive(Debug, Clone, Copy)]
struct Q4BlockMetadata {
    index: usize,
    scale_bits: u16,
    bias_bits: u16,
}

#[derive(Debug, Clone, Copy)]
enum IngressErrorKind {
    InvalidSafetensors,
    InvalidInput,
}

impl IngressErrorKind {
    fn error(self, message: String) -> InferenceError {
        match self {
            Self::InvalidSafetensors => InferenceError::InvalidSafetensors(message),
            Self::InvalidInput => InferenceError::InvalidInput(message),
        }
    }
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
            expected_shape: None,
            payload: IngestPayload::DecodedF32 {
                values,
                dtype_label,
                error_kind: IngressErrorKind::InvalidSafetensors,
            },
        }
    }

    /// Wrap the decoded f32 source of an in-memory Q8 conversion.
    pub(crate) fn q8_source(
        source: &'a str,
        tensor_name: &'a str,
        shape: &'a [usize],
        values: &'a [f32],
    ) -> Self {
        Self {
            source,
            tensor_name,
            shape,
            expected_shape: None,
            payload: IngestPayload::DecodedF32 {
                values,
                dtype_label: "F32",
                error_kind: IngressErrorKind::InvalidInput,
            },
        }
    }

    /// Wrap derived Q8 matrix data and scales for validation.
    pub(crate) fn q8(
        source: &'a str,
        tensor_name: &'a str,
        shape: &'a [usize],
        data: &'a [i8],
        scales: &'a [f32],
    ) -> Self {
        Self {
            source,
            tensor_name,
            shape,
            expected_shape: None,
            payload: IngestPayload::Q8 { data, scales },
        }
    }

    /// Wrap native Q4 layout and optional per-block metadata for validation.
    pub(crate) fn native_q4(
        source: &'a str,
        tensor_name: &'a str,
        shape: &'a [usize],
        original_len: usize,
        block_count: usize,
    ) -> Self {
        Self {
            source,
            tensor_name,
            shape,
            expected_shape: None,
            payload: IngestPayload::NativeQ4 {
                original_len,
                block_count,
            },
        }
    }

    /// Wrap one native Q4 block's serialized metadata for validation.
    pub(crate) fn native_q4_block(
        source: &'a str,
        tensor_name: &'a str,
        index: usize,
        scale_bits: u16,
        bias_bits: u16,
    ) -> Self {
        Self {
            source,
            tensor_name,
            shape: &[],
            expected_shape: None,
            payload: IngestPayload::NativeQ4Block(Q4BlockMetadata {
                index,
                scale_bits,
                bias_bits,
            }),
        }
    }

    /// Require the declared tensor shape to match model-config geometry.
    pub(crate) fn with_expected_shape(mut self, expected_shape: &'a [usize]) -> Self {
        self.expected_shape = Some(expected_shape);
        self
    }
}

fn checked_shape_numel(tensor: &IngestedTensor<'_>) -> Result<usize, InferenceError> {
    tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "{}: tensor {} shape {:?} overflows usize element count",
                tensor.source, tensor.tensor_name, tensor.shape,
            ))
        })
}

/// Validate one ingested tensor at the exact point its bytes become trusted
/// tensor data: the declared shape's element count must not overflow, the
/// payload's element count must exactly match that shape, and every element
/// must be finite (NaN and either infinity are rejected; signed zero and
/// subnormals are accepted).
///
/// Safetensors failures use `InferenceError::InvalidSafetensors`; in-memory
/// conversion failures use `InferenceError::InvalidInput`. Both carry source
/// label, tensor name, dtype, shape, and (for a finite-value failure) the
/// first offending element index.
pub(crate) fn validate_ingested_tensor(tensor: IngestedTensor<'_>) -> Result<(), InferenceError> {
    if let Some(expected_shape) = tensor.expected_shape
        && tensor.shape != expected_shape
    {
        return Err(InferenceError::InvalidSafetensors(format!(
            "{}: tensor {} has shape {:?}, expected {:?}",
            tensor.source, tensor.tensor_name, tensor.shape, expected_shape,
        )));
    }

    match tensor.payload {
        IngestPayload::DecodedF32 {
            values,
            dtype_label,
            error_kind,
        } => {
            let numel = checked_shape_numel(&tensor)?;
            if values.len() != numel {
                return Err(error_kind.error(format!(
                    "{}: tensor {} ({dtype_label}) decoded element count {} does not match \
                     shape {:?} (shape product {numel})",
                    tensor.source,
                    tensor.tensor_name,
                    values.len(),
                    tensor.shape,
                )));
            }
            if let Some((idx, bad)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(error_kind.error(format!(
                    "{}: tensor {} ({dtype_label}) has non-finite value {bad} at element index \
                     {idx} of {numel} (shape {:?})",
                    tensor.source, tensor.tensor_name, tensor.shape,
                )));
            }
            Ok(())
        }
        IngestPayload::Q8 { data, scales } => {
            if tensor.shape.len() != 2 {
                return Err(InferenceError::InvalidInput(format!(
                    "{}: tensor {} (Q8) shape {:?} must have exactly two dimensions",
                    tensor.source, tensor.tensor_name, tensor.shape,
                )));
            }
            let numel = checked_shape_numel(&tensor)?;
            if data.len() != numel {
                return Err(InferenceError::InvalidInput(format!(
                    "{}: tensor {} (Q8) data element count {} does not match shape {:?} \
                     (expected {numel})",
                    tensor.source,
                    tensor.tensor_name,
                    data.len(),
                    tensor.shape,
                )));
            }

            let expected_scales = tensor.shape[0];
            if scales.len() != expected_scales {
                return Err(InferenceError::InvalidInput(format!(
                    "{}: tensor {} (Q8) scale count {} does not match row count \
                     {expected_scales} (shape {:?})",
                    tensor.source,
                    tensor.tensor_name,
                    scales.len(),
                    tensor.shape,
                )));
            }
            // Single pass over `scales`: both the non-finite and non-positive checks below
            // scan the same per-row scale vector, so fusing them avoids a second full read.
            // Priority is preserved exactly — non-finite is reported before non-positive when
            // a row somehow triggers both (e.g. `-inf`, which `is_finite()` rejects but
            // `<= 0.0` also matches) — by resolving `first_non_finite` first below.
            let mut first_non_finite: Option<(usize, f32)> = None;
            let mut first_non_positive: Option<(usize, f32)> = None;
            for (row, &scale) in scales.iter().enumerate() {
                if first_non_finite.is_none() && !scale.is_finite() {
                    first_non_finite = Some((row, scale));
                }
                if first_non_positive.is_none() && scale <= 0.0 {
                    first_non_positive = Some((row, scale));
                }
            }
            if let Some((row, scale)) = first_non_finite {
                return Err(InferenceError::InvalidInput(format!(
                    "{}: tensor {} (Q8) has non-finite scale {scale} at row {row} of \
                     {expected_scales} (shape {:?})",
                    tensor.source, tensor.tensor_name, tensor.shape,
                )));
            }
            if let Some((row, scale)) = first_non_positive {
                return Err(InferenceError::InvalidInput(format!(
                    "{}: tensor {} (Q8) has non-positive scale {scale} at row {row} of \
                     {expected_scales} (shape {:?})",
                    tensor.source, tensor.tensor_name, tensor.shape,
                )));
            }
            Ok(())
        }
        IngestPayload::NativeQ4 {
            original_len,
            block_count,
        } => {
            let numel = checked_shape_numel(&tensor)?;
            if original_len != numel {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} (Q4) original_len {original_len} does not match shape {:?} \
                     (expected {numel})",
                    tensor.source, tensor.tensor_name, tensor.shape,
                )));
            }
            let expected_blocks = original_len.div_ceil(super::q4_weights::Q4_BLOCK_WEIGHTS);
            if block_count != expected_blocks {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} (Q4) block count {block_count} does not match original_len \
                     {original_len} (expected {expected_blocks})",
                    tensor.source, tensor.tensor_name,
                )));
            }
            Ok(())
        }
        IngestPayload::NativeQ4Block(metadata) => {
            let scale = super::half_bits::f16_bits_to_f32(metadata.scale_bits);
            let bias = super::half_bits::f16_bits_to_f32(metadata.bias_bits);
            if !scale.is_finite() || scale <= 0.0 {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} (Q4) block {} has invalid scale {scale}; scale must be \
                     finite and strictly positive",
                    tensor.source, tensor.tensor_name, metadata.index,
                )));
            }
            if !bias.is_finite() {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "{}: tensor {} (Q4) block {} has non-finite bias {bias}",
                    tensor.source, tensor.tensor_name, metadata.index,
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

    #[test]
    fn q8_rejects_non_finite_source_with_tensor_attribution() {
        let source_values = [1.0f32, 2.0, f32::NAN, 4.0];
        let tensor = IngestedTensor::q8_source(
            "in-memory Q8 quantization",
            "q_proj",
            &[2, 2],
            &source_values,
        );

        let err = validate_ingested_tensor(tensor).expect_err("NaN must be rejected");
        match err {
            InferenceError::InvalidInput(msg) => {
                assert!(
                    msg.contains("q_proj"),
                    "error should name the tensor: {msg}"
                );
                assert!(
                    msg.contains("element index 2"),
                    "error should point at the offending source value: {msg}"
                );
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn q8_rejects_quantized_geometry_mismatch() {
        let data = [1i8, 2, 3];
        let scales = [0.5f32, 1.0];
        let tensor = IngestedTensor::q8(
            "in-memory Q8 quantization",
            "q_proj",
            &[2, 2],
            &data,
            &scales,
        );

        let err = validate_ingested_tensor(tensor).expect_err("short Q8 data must be rejected");
        assert!(err.to_string().contains("data element count 3"));

        let data = [1i8, 2, 3, 4];
        let scales = [0.5f32];
        let tensor = IngestedTensor::q8(
            "in-memory Q8 quantization",
            "q_proj",
            &[2, 2],
            &data,
            &scales,
        );
        let err = validate_ingested_tensor(tensor).expect_err("short scales must be rejected");
        assert!(err.to_string().contains("scale count 1"));
    }

    #[test]
    fn q8_rejects_non_positive_scale() {
        let data = [1i8, 2, 3, 4];

        for bad in [0.0f32, -1.0] {
            let scales = [0.5f32, bad];
            let tensor = IngestedTensor::q8(
                "in-memory Q8 quantization",
                "q_proj",
                &[2, 2],
                &data,
                &scales,
            );

            let err = validate_ingested_tensor(tensor)
                .expect_err("non-positive Q8 scale must be rejected");
            assert!(
                err.to_string().contains("non-positive scale"),
                "unexpected error for scale {bad}: {err}"
            );
        }
    }

    #[test]
    fn q8_rejects_non_finite_scale() {
        let data = [1i8, 2, 3, 4];
        let scales = [0.5f32, f32::INFINITY];
        let tensor = IngestedTensor::q8(
            "in-memory Q8 quantization",
            "q_proj",
            &[2, 2],
            &data,
            &scales,
        );

        let err =
            validate_ingested_tensor(tensor).expect_err("non-finite Q8 scale must be rejected");
        assert!(err.to_string().contains("non-finite scale"));
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
