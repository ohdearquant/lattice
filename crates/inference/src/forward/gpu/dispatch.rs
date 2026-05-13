use super::api::Result;
use super::dims::ForwardDims;
use super::params::{
    DispatchParams, ELEM_WG, P_EPS, P_GROUPS, P_HALF_DIM, P_HEAD_DIM, P_K, P_M, P_N, P_NUM_HEADS,
    P_NUM_KV_HEADS, P_NUM_ROWS, P_ROW_LEN, P_SCALE, P_SEQ_LEN, P_TOTAL_ELEMS, ParamPacker, ROPE_WG,
    TILE,
};
use super::util::ceil_div_u32;

pub(super) fn dispatch_matmul(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    m: u32,
    n: u32,
    k: u32,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_M, m);
    p.set_u32(P_N, n);
    p.set_u32(P_K, k);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(ceil_div_u32(n, TILE), ceil_div_u32(m, TILE), 1);
    Ok(())
}

pub(super) fn dispatch_rms_norm(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    num_rows: u32,
    row_len: u32,
    eps: f32,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_ROW_LEN, row_len);
    p.set_u32(P_NUM_ROWS, num_rows);
    p.set_f32(P_EPS, eps);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(num_rows, 1, 1);
    Ok(())
}

pub(super) fn dispatch_copy(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    total_elems: u32,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_TOTAL_ELEMS, total_elems);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(ceil_div_u32(total_elems, ELEM_WG), 1, 1);
    Ok(())
}

pub(super) fn dispatch_add(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    total_elems: u32,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_TOTAL_ELEMS, total_elems);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(ceil_div_u32(total_elems, ELEM_WG), 1, 1);
    Ok(())
}

pub(super) fn dispatch_silu(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    total_elems: u32,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_TOTAL_ELEMS, total_elems);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(ceil_div_u32(total_elems, ELEM_WG), 1, 1);
    Ok(())
}

pub(super) fn dispatch_mul(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    total_elems: u32,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_TOTAL_ELEMS, total_elems);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(ceil_div_u32(total_elems, ELEM_WG), 1, 1);
    Ok(())
}

pub(super) fn dispatch_rope(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_SEQ_LEN, seq_len);
    p.set_u32(P_NUM_HEADS, num_heads);
    p.set_u32(P_HEAD_DIM, head_dim);
    p.set_u32(P_HALF_DIM, head_dim / 2);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(ceil_div_u32(head_dim / 2, ROPE_WG), seq_len, num_heads);
    Ok(())
}

pub(super) fn dispatch_attention_scores(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dims: &ForwardDims,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_SEQ_LEN, dims.seq);
    p.set_u32(P_NUM_HEADS, dims.num_heads);
    p.set_u32(P_NUM_KV_HEADS, dims.num_kv_heads);
    p.set_u32(P_HEAD_DIM, dims.head_dim);
    p.set_u32(P_GROUPS, dims.groups);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(
        ceil_div_u32(dims.seq, TILE),
        ceil_div_u32(dims.seq, TILE),
        dims.num_heads,
    );
    Ok(())
}

pub(super) fn dispatch_attention_softmax(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dims: &ForwardDims,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_SEQ_LEN, dims.seq);
    p.set_u32(P_NUM_HEADS, dims.num_heads);
    p.set_f32(P_SCALE, 1.0 / (dims.head_dim as f32).sqrt());
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(dims.seq, dims.num_heads, 1);
    Ok(())
}

pub(super) fn dispatch_attention_context(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dims: &ForwardDims,
    params: &mut ParamPacker,
) -> Result<()> {
    let mut p = DispatchParams::zeroed();
    p.set_u32(P_SEQ_LEN, dims.seq);
    p.set_u32(P_NUM_HEADS, dims.num_heads);
    p.set_u32(P_NUM_KV_HEADS, dims.num_kv_heads);
    p.set_u32(P_HEAD_DIM, dims.head_dim);
    p.set_u32(P_GROUPS, dims.groups);
    let offset = params.push(p)?;
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[offset]);
    pass.dispatch_workgroups(
        ceil_div_u32(dims.head_dim, TILE),
        ceil_div_u32(dims.seq, TILE),
        dims.num_heads,
    );
    Ok(())
}
