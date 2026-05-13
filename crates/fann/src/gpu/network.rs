//! GPU-accelerated neural network inference

use super::context::GpuContext;
use super::error::{GpuError, GpuResult};
use super::shader_manager::ShaderType;
use super::{should_use_gpu, thresholds};
use crate::Network;
use crate::activation::Activation;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Uniform buffer for matrix operations
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MatmulUniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform buffer for activation functions
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ActivationUniforms {
    size: u32,
    alpha: f32, // For LeakyReLU
    _pad0: u32,
    _pad1: u32,
}

/// Layer data stored on GPU
struct GpuLayer {
    weight_buffer: wgpu::Buffer,
    bias_buffer: wgpu::Buffer,
    num_inputs: usize,
    num_outputs: usize,
    activation: Activation,
}

/// GPU-accelerated neural network
///
/// Wraps a CPU Network and provides GPU-accelerated inference
/// with automatic CPU fallback for small operations.
pub struct GpuNetwork {
    ctx: Arc<GpuContext>,
    /// Original CPU network (kept for fallback and weight access)
    cpu_network: Network,
    /// Layer data on GPU
    layers: Vec<GpuLayer>,
    /// Whether to use GPU (based on network size)
    use_gpu: bool,
}

impl GpuNetwork {
    /// Create a GPU-accelerated network from a CPU network
    pub fn new(ctx: Arc<GpuContext>, network: Network) -> GpuResult<Self> {
        // Determine if GPU is worth using for this network size
        let total_params = network.total_params();
        let use_gpu = should_use_gpu(total_params, 1);

        // Upload layer data to GPU
        let mut layers = Vec::with_capacity(network.num_layers());
        for layer in network.layers() {
            let weight_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("weights"),
                    contents: bytemuck::cast_slice(layer.weights()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

            let bias_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("biases"),
                    contents: bytemuck::cast_slice(layer.biases()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

            layers.push(GpuLayer {
                weight_buffer,
                bias_buffer,
                num_inputs: layer.num_inputs(),
                num_outputs: layer.num_outputs(),
                activation: layer.activation(),
            });
        }

        Ok(Self {
            ctx,
            cpu_network: network,
            layers,
            use_gpu,
        })
    }

    /// Run forward pass
    ///
    /// Automatically uses GPU or CPU based on operation size.
    pub async fn forward(&mut self, input: &[f32]) -> GpuResult<Vec<f32>> {
        if input.len() != self.cpu_network.num_inputs() {
            return Err(GpuError::InvalidDimensions(format!(
                "Expected {} inputs, got {}",
                self.cpu_network.num_inputs(),
                input.len()
            )));
        }

        // Use CPU for small networks
        if !self.use_gpu {
            return self.forward_cpu(input);
        }

        self.forward_gpu(input).await
    }

    /// Synchronous forward pass
    pub fn forward_sync(&mut self, input: &[f32]) -> GpuResult<Vec<f32>> {
        pollster::block_on(self.forward(input))
    }

    /// GPU forward pass
    async fn forward_gpu(&mut self, input: &[f32]) -> GpuResult<Vec<f32>> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Create input buffer
        let mut current_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let mut current_size = input.len();

        // Process each layer
        for layer in self.layers.iter() {
            // Check watchdog limit
            let elements = layer.num_inputs * layer.num_outputs;
            if elements > thresholds::MAX_ELEMENTS_PER_DISPATCH {
                // Fall back to CPU for this layer to avoid watchdog
                // In production, we'd tile the dispatch instead
                return self.forward_cpu(input);
            }

            // Determine which shader to use (fused if ReLU)
            let shader_type = if matches!(layer.activation, Activation::ReLU) {
                ShaderType::MatrixVectorMultiplyRelu
            } else {
                ShaderType::MatrixVectorMultiply
            };

            let pipeline = self.ctx.shader_manager.get_or_compile(shader_type)?;

            // Create output buffer
            let output_size = layer.num_outputs * std::mem::size_of::<f32>();
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("output"),
                size: output_size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create uniforms
            let uniforms = MatmulUniforms {
                rows: layer.num_outputs as u32,
                cols: layer.num_inputs as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Create bind group
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: layer.weight_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: current_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: layer.bias_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

            // Dispatch compute
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                let workgroup_size = shader_type.workgroup_size()[0];
                let workgroups = (layer.num_outputs as u32).div_ceil(workgroup_size);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // Apply non-fused activations
            if !matches!(layer.activation, Activation::ReLU | Activation::Linear) {
                self.apply_activation(&output_buffer, layer.num_outputs, layer.activation)?;
            }

            current_buffer = output_buffer;
            current_size = layer.num_outputs;
        }

        // Read back results
        self.read_buffer(&current_buffer, current_size).await
    }

    /// Apply activation function on GPU
    fn apply_activation(
        &self,
        buffer: &wgpu::Buffer,
        size: usize,
        activation: Activation,
    ) -> GpuResult<()> {
        let shader_type = match activation {
            Activation::ReLU => ShaderType::ReLU,
            Activation::LeakyReLU(_) => ShaderType::LeakyReLU,
            Activation::Sigmoid => ShaderType::Sigmoid,
            Activation::Tanh => ShaderType::Tanh,
            Activation::Linear => return Ok(()), // No-op
            Activation::Softmax => {
                // Softmax is only applied to the final output layer on CPU.
                // Non-final softmax layers produce incorrect GPU results.
                // FIXME(FANN-M5): wire per-layer CPU softmax fallback.
                return Ok(());
            }
        };

        let pipeline = self.ctx.shader_manager.get_or_compile(shader_type)?;

        let alpha = match activation {
            Activation::LeakyReLU(a) => a,
            _ => 0.0,
        };

        let uniforms = ActivationUniforms {
            size: size as u32,
            alpha,
            _pad0: 0,
            _pad1: 0,
        };

        let uniform_buffer =
            self.ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("activation_uniforms"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("activation_bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = shader_type.workgroup_size()[0];
            let workgroups = (size as u32).div_ceil(workgroup_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Read buffer contents back to CPU
    async fn read_buffer(&self, buffer: &wgpu::Buffer, size: usize) -> GpuResult<Vec<f32>> {
        let staging_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.ctx.device.poll(wgpu::Maintain::Wait);

        receiver
            .await
            .map_err(|_| GpuError::BufferRead("Channel closed".into()))?
            .map_err(|e| GpuError::BufferRead(format!("{e:?}")))?;

        let data = buffer_slice.get_mapped_range();
        let mut output: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Apply softmax on CPU if last layer uses it
        if let Some(layer) = self.layers.last() {
            if matches!(layer.activation, Activation::Softmax) {
                let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for x in output.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for x in output.iter_mut() {
                    *x /= sum;
                }
            }
        }

        Ok(output)
    }

    /// CPU forward pass (fallback)
    fn forward_cpu(&mut self, input: &[f32]) -> GpuResult<Vec<f32>> {
        self.cpu_network
            .forward(input)
            .map(|s| s.to_vec())
            .map_err(|e| GpuError::Network(e.to_string()))
    }

    /// Get reference to CPU network
    pub fn cpu_network(&self) -> &Network {
        &self.cpu_network
    }

    /// Sync weights from CPU to GPU
    pub fn sync_weights(&mut self) -> GpuResult<()> {
        for (layer_data, cpu_layer) in self.layers.iter().zip(self.cpu_network.layers()) {
            self.ctx.queue.write_buffer(
                &layer_data.weight_buffer,
                0,
                bytemuck::cast_slice(cpu_layer.weights()),
            );
            self.ctx.queue.write_buffer(
                &layer_data.bias_buffer,
                0,
                bytemuck::cast_slice(cpu_layer.biases()),
            );
        }
        Ok(())
    }

    /// Check if using GPU
    pub fn is_using_gpu(&self) -> bool {
        self.use_gpu
    }

    /// Flush GPU memory
    ///
    /// Releases cached buffers and waits for pending GPU operations.
    /// Call this periodically during long training loops to prevent OOM
    /// from async VRAM deallocation lag.
    ///
    /// Returns the number of bytes freed from the buffer pool.
    pub fn flush(&self) -> u64 {
        self.ctx.flush_memory()
    }

    /// Get GPU context reference
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkBuilder;

    fn skip_if_no_gpu() -> bool {
        !super::super::is_gpu_available()
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
    fn test_gpu_network_creation() {
        if skip_if_no_gpu() {
            println!("Skipping GPU test - no GPU available");
            return;
        }

        let ctx = Arc::new(GpuContext::new_blocking().unwrap());
        let network = NetworkBuilder::new()
            .input(4)
            .hidden(8, Activation::ReLU)
            .output(2, Activation::Softmax)
            .build()
            .unwrap();

        let gpu_net = GpuNetwork::new(ctx, network);
        assert!(gpu_net.is_ok());
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
    fn test_gpu_forward() {
        if skip_if_no_gpu() {
            return;
        }

        let ctx = Arc::new(GpuContext::new_blocking().unwrap());
        let network = NetworkBuilder::new()
            .input(4)
            .hidden(8, Activation::ReLU)
            .output(2, Activation::Softmax)
            .build()
            .unwrap();

        let mut gpu_net = GpuNetwork::new(ctx, network).unwrap();

        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = gpu_net.forward_sync(&input).unwrap();

        assert_eq!(output.len(), 2);
        // Softmax should sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
