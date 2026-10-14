#include <metal_stdlib>
using namespace metal;

// Axial (2-D) rotary position embedding for the EmbeddingGemma 2 vision tower.
//
// Each head is split into two head_dim/2 halves: the first rotates by the axis-0 (x) angle and
// the second by the axis-1 (y) angle. Within a half, channel j pairs with channel
// j + head_dim/4. The inverse-frequency table holds the axis-0 block first, then the axis-1
// block, head_dim/4 entries each; `positions` holds two floats per token (x then y). This mirrors
// `vision_axial_rope_apply` pair for pair; the whole-head stride-half pairing used by the text
// towers is a different map and must not be substituted here.
kernel void eg2_vision_axial_rope(
    device float* x               [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device const float* inv_freq   [[buffer(2)]],
    constant uint& num_tokens      [[buffer(3)]],
    constant uint& num_heads       [[buffer(4)]],
    constant uint& head_dim        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint per_axis = head_dim / 4u;
    const uint part = head_dim / 2u;
    const uint pairs_per_head = head_dim / 2u;
    const uint total_pairs = num_tokens * num_heads * pairs_per_head;
    if (gid >= total_pairs) return;

    const uint pair = gid % pairs_per_head;
    const uint head = (gid / pairs_per_head) % num_heads;
    const uint token = gid / (num_heads * pairs_per_head);
    const uint axis = pair / per_axis;
    const uint j = pair % per_axis;

    const float coordinate = positions[2u * token + axis];
    const float angle = coordinate * inv_freq[pair];
    const float s = precise::sin(angle);
    const float c = precise::cos(angle);

    const uint offset = (token * num_heads + head) * head_dim + axis * part;
    const float lo = x[offset + j];
    const float hi = x[offset + per_axis + j];
    x[offset + j] = lo * c - hi * s;
    x[offset + per_axis + j] = hi * c + lo * s;
}
