//! Compile-only compatibility witnesses for compute-owned forward kernels.

use lattice_compute::forward as owner;
use lattice_inference::forward as facade;

type Packing =
    fn(&[f32], usize, usize) -> Result<(Vec<u8>, Vec<f32>), lattice_inference_core::InferenceError>;
type Matvec = fn(&[i8], f32, &[u8], &[f32], usize, usize, &mut [f32]);
type TernaryMatmul = fn(&[f32], &[u8], &[f32], usize, usize) -> Vec<f32>;
#[cfg(target_arch = "aarch64")]
type NeonMatvec = unsafe fn(&[i8], f32, &[u8], &[f32], usize, usize, &mut [f32]);
type GpuMatmul = fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> bool;

const _: fn(usize) -> usize = facade::bitnet_kernel::packed_row_bytes;
const _: fn(usize) -> usize = owner::bitnet_kernel::packed_row_bytes;
const _: Packing = facade::bitnet_kernel::pack_ternary;
const _: Packing = owner::bitnet_kernel::pack_ternary;
const _: fn(&[u8], usize, usize, usize) -> i8 = facade::bitnet_kernel::unpack_weight;
const _: fn(&[u8], usize, usize, usize) -> i8 = owner::bitnet_kernel::unpack_weight;
const _: fn(&[f32]) -> (Vec<i8>, f32) = facade::bitnet_kernel::quantize_activation;
const _: fn(&[f32]) -> (Vec<i8>, f32) = owner::bitnet_kernel::quantize_activation;
const _: Matvec = facade::bitnet_kernel::matvec_ternary_scalar;
const _: Matvec = owner::bitnet_kernel::matvec_ternary_scalar;
const _: TernaryMatmul = facade::bitnet_kernel::matmul_ternary;
const _: TernaryMatmul = owner::bitnet_kernel::matmul_ternary;
#[cfg(target_arch = "aarch64")]
const _: NeonMatvec = facade::bitnet_kernel::matvec_ternary_neon;
#[cfg(target_arch = "aarch64")]
const _: NeonMatvec = owner::bitnet_kernel::matvec_ternary_neon;
const _: GpuMatmul = facade::gpu_gemm::gpu_matmul;
const _: GpuMatmul = owner::gpu_gemm::gpu_matmul;
const _: GpuMatmul = facade::gpu_gemm::gpu_matmul_bt;
const _: GpuMatmul = owner::gpu_gemm::gpu_matmul_bt;
