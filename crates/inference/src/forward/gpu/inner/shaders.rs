//! WGSL shader source constants for matmul, RMSNorm, copy, add, and related GPU kernels.
pub(super) const MATMUL_BT_SHADER: &str = concat!("\n", include_str!("../../wgsl/matmul_bt.wgsl"));

pub(super) const RMS_NORM_SHADER: &str = concat!("\n", include_str!("../../wgsl/rms_norm.wgsl"));

pub(super) const COPY_SHADER: &str = concat!("\n", include_str!("../../wgsl/copy.wgsl"));

pub(super) const ADD_SHADER: &str = concat!("\n", include_str!("../../wgsl/add.wgsl"));

pub(super) const SILU_SHADER: &str = concat!("\n", include_str!("../../wgsl/silu.wgsl"));

pub(super) const MUL_SHADER: &str = concat!("\n", include_str!("../../wgsl/mul.wgsl"));

pub(super) const ROPE_SHADER: &str = concat!("\n", include_str!("../../wgsl/rope.wgsl"));

pub(super) const ATTENTION_SCORES_SHADER: &str =
    concat!("\n", include_str!("../../wgsl/attention_scores.wgsl"));

// Guarantee scope (#795): the fail-closed
// zero-row contract below has been verified on this repository's native
// Metal-backed WGPU adapter (`GpuModelState::new`'s
// `wgpu::PowerPreference::HighPerformance` request, backend `Backends::all()`
// resolving to Metal on macOS CI/dev hosts). It is NOT a WGSL-portable
// guarantee: WGSL's Finite Math Assumption
// (https://www.w3.org/TR/WGSL/#finite-math-assumption) permits an
// implementation to assume NaN and infinities are absent during shader
// execution, so a different backend (Vulkan/DX12/browser WebGPU) may
// legally replace a poisoned value with an indeterminate finite-looking one
// at any point after it participates in WGSL floating-point arithmetic,
// before `is_non_finite`'s bitcast ever inspects it. Reading the raw score
// before the `* scale` multiply (below) narrows the window but cannot close
// it for backends whose compiler chooses to optimize on the assumption
// upstream of that read. The claim this kernel makes is therefore: "fails
// closed on the tested native backend". The CPU parity / differential
// (`e2e-parity.yml`, HF-vs-lattice greedy-token) and CPU-side
// `attention::softmax_row` gates define the reference semantics for future
// backend-specific differential coverage; detecting a defeated guard on an
// untested WGPU backend requires adding that backend to a differential run —
// those gates do not execute this shader on Vulkan/DX12/browser WebGPU today.
pub(super) const ATTENTION_SOFTMAX_SHADER: &str =
    concat!("\n", include_str!("../../wgsl/attention_softmax.wgsl"));

pub(super) const ATTENTION_CONTEXT_SHADER: &str =
    concat!("\n", include_str!("../../wgsl/attention_context.wgsl"));
