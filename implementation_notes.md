# Implementation Notes

- Extended the private weight-ingress seam to decode QuaRot F32/F16/BF16 bytes into `f64` while checking finiteness in the same materialization pass.
- Routed both offline quantizers through one checked KHF1 writer, removing the plain quantizer's local half encoder and validating f16 output before file creation.
- Rejected Q4 scale or bias metadata that becomes non-finite in f16, and changed Q4 shape mismatches from panics to `InferenceError::InvalidInput` results.
- Added regression coverage for NaN and both infinities across all supported QuaRot source dtypes, invalid-output publication, f16 overflow, Q4 metadata overflow, and shape errors.
