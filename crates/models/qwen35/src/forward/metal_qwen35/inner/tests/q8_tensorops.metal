#include <metal_stdlib>

#if !defined(__METAL_VERSION__) || __METAL_VERSION__ < 400
#error "The Q8 TensorOps probe requires Metal 4.0 or newer"
#endif

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#if !defined(__HAVE_TENSOR__)
#error "The Q8 TensorOps probe requires tensor support"
#endif

using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemm_q8_tensorops_probe(
    device const char* qw [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& y_stride [[buffer(6)]],
    device float* markers [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile = 32;
    constexpr uint threads = 128;
    const uint row_start = group.y * tile;
    const uint col_start = group.x * tile;
    const int valid_m = int(min(tile, m - row_start));
    const int valid_n = int(min(tile, n - col_start));
    const uint row_bytes = (k / 32) * 34;
    threadgroup half stage_a[tile * tile];
    threadgroup half stage_b[tile * tile];
    auto a = tensor(stage_a, dextents<int, 2>{32, valid_m}, array<int, 2>{1, 32});
    auto b = tensor(stage_b, dextents<int, 2>{valid_n, 32}, array<int, 2>{1, 32});
    auto output = tensor(y + row_start * y_stride + col_start,
                         dextents<int, 2>{valid_n, valid_m},
                         array<int, 2>{1, int(y_stride)});
    constexpr auto descriptor = matmul2d_descriptor(
        32, 32, 32, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(a), decltype(b), float>();
    for (uint16_t i = 0; i < accumulator.get_capacity(); ++i) {
        accumulator.set(i, 0.0f);
    }

    for (uint k_start = 0; k_start < k; k_start += tile) {
        for (uint index = tid; index < tile * tile; index += threads) {
            const uint row = index / tile;
            const uint col = index % tile;
            // Poison unused storage so partial tiles rely on dynamic tensor bounds.
            stage_a[index] = row < uint(valid_m)
                ? half(x[(row_start + row) * k + k_start + col]) : half(777.0f);
            half weight = half(777.0f);
            if (col < uint(valid_n)) {
                const uint block = (col_start + col) * row_bytes + (k_start / 32) * 34;
                const ushort scale_bits = ushort(uchar(qw[block]))
                    | (ushort(uchar(qw[block + 1])) << 8);
                const float scale = float(as_type<half>(scale_bits));
                weight = half(float(qw[block + 2 + row]) * scale);
            }
            stage_b[index] = weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        operation.run(a, b, accumulator);
        // All four SIMD groups finish reading before any staging entry is recycled.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    accumulator.store(output);
    if (tid == 0) {
        const uint marker = 2 * (group.y * ((n + tile - 1) / tile) + group.x);
        markers[marker] = float(__METAL_VERSION__);
        markers[marker + 1] = 268.0f;
    }
}
