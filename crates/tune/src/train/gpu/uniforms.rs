//! GPU shader uniform structures

/// Uniforms for Adam optimizer shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamUniforms {
    pub size: u32,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Uniforms for AdamW optimizer shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamWUniforms {
    pub size: u32,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub t: f32,
    pub _pad: u32,
}

/// Uniforms for SGD momentum shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SgdMomentumUniforms {
    pub size: u32,
    pub learning_rate: f32,
    pub momentum: f32,
    pub _pad: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_uniforms_size() {
        // Verify struct is properly aligned for GPU
        assert_eq!(std::mem::size_of::<AdamUniforms>(), 32);
    }

    #[test]
    fn test_adamw_uniforms_size() {
        assert_eq!(std::mem::size_of::<AdamWUniforms>(), 32);
    }

    #[test]
    fn test_sgd_momentum_uniforms_size() {
        assert_eq!(std::mem::size_of::<SgdMomentumUniforms>(), 16);
    }
}
