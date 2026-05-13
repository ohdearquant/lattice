//! GPU buffer state management

/// GPU buffer for optimizer state
pub struct OptimizerState {
    /// First moment (Adam)
    pub m: wgpu::Buffer,
    /// Second moment (Adam)
    pub v: wgpu::Buffer,
    /// Velocity (SGD momentum)
    pub velocity: wgpu::Buffer,
    /// Current timestep
    pub t: u32,
}

/// Layer gradients stored on GPU
pub struct LayerGradients {
    pub weight_grads: wgpu::Buffer,
    pub bias_grads: wgpu::Buffer,
    pub optimizer_state: OptimizerState,
    pub num_weights: usize,
    pub num_biases: usize,
}
