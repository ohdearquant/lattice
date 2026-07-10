//! GPU buffer state management

/// GPU buffer for optimizer state
///
/// `m`, `v`, and `velocity` are pre-allocated by `GpuTrainer::init_gradients`
/// but not yet bound to any shader dispatch: the GPU-shader optimizer arms
/// (Adam, AdamW, SGD-momentum) fail loudly instead of running until the real
/// buffer bindings are wired (#797). Kept as placeholders for that follow-up.
pub struct OptimizerState {
    /// First moment (Adam)
    #[allow(dead_code, reason = "pre-allocated for #797 follow-up buffer wiring")]
    pub m: wgpu::Buffer,
    /// Second moment (Adam)
    #[allow(dead_code, reason = "pre-allocated for #797 follow-up buffer wiring")]
    pub v: wgpu::Buffer,
    /// Velocity (SGD momentum)
    #[allow(dead_code, reason = "pre-allocated for #797 follow-up buffer wiring")]
    pub velocity: wgpu::Buffer,
    /// Current timestep
    pub t: u32,
}

/// Layer gradients stored on GPU
pub struct LayerGradients {
    pub weight_grads: wgpu::Buffer,
    pub bias_grads: wgpu::Buffer,
    pub optimizer_state: OptimizerState,
    #[allow(dead_code, reason = "pre-allocated for #797 follow-up buffer wiring")]
    pub num_weights: usize,
    #[allow(dead_code, reason = "pre-allocated for #797 follow-up buffer wiring")]
    pub num_biases: usize,
}
