//! Compile-only witnesses for the Qwen3 WGPU compatibility surface.

#[cfg(feature = "wgpu-gpu")]
mod enabled {
    use lattice_inference::forward::gpu as facade;
    use lattice_model_qwen3_embedding::gpu as owner;

    const _: fn(facade::GpuForwardError) -> owner::GpuForwardError = |value| value;
    const _: fn(facade::GpuModelState) -> owner::GpuModelState = |value| value;
    const _: fn(facade::GpuRuntimeConfig) -> owner::GpuRuntimeConfig = |value| value;
    const _: fn(facade::Qwen3Config) -> owner::Qwen3Config = |value| value;
    const _: fn(facade::Qwen3LayerWeights) -> owner::Qwen3LayerWeights = |value| value;
    const _: fn(facade::Qwen3Weights) -> owner::Qwen3Weights = |value| value;

    const _: fn() = || {
        let _ = facade::Qwen3Config::q_dim;
        let _ = facade::Qwen3Config::kv_dim;
        let _ = facade::Qwen3Config::num_groups;
        let _ = facade::Qwen3Weights::validate;
        let _ = facade::GpuModelState::new;
        let _ = facade::GpuModelState::forward;
        let _ = facade::GpuModelState::submit_count;
        let _ = facade::GpuModelState::user_buffer_creation_count;
        let _ = facade::GpuModelState::has_gpu_embedding_buffer;
    };
}
