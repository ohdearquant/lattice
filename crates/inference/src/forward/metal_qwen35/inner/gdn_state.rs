use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetalGdnStatePrecision {
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MetalGdnStateGeometry {
    pub(crate) architectural_layers: usize,
    pub(crate) active_layers: usize,
    pub(crate) allocated_layers: usize,
    pub(crate) qkv_dim: usize,
    pub(crate) conv_history: usize,
    pub(crate) value_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_dim: usize,
}

impl MetalGdnStateGeometry {
    pub(crate) fn conv_elements_per_layer(self) -> usize {
        self.qkv_dim * self.conv_history
    }

    pub(crate) fn matrix_elements_per_layer(self) -> usize {
        self.value_heads * self.key_dim * self.value_dim
    }
}

pub(crate) struct MetalGdnLayerState {
    conv_buffer: Buffer,
    s_matrix: Buffer,
}

impl MetalGdnLayerState {
    pub(crate) fn conv_buffer(&self) -> &Buffer {
        &self.conv_buffer
    }

    pub(crate) fn s_matrix(&self) -> &Buffer {
        &self.s_matrix
    }
}

pub(crate) struct MetalGdnState {
    geometry: MetalGdnStateGeometry,
    precision: MetalGdnStatePrecision,
    layers: Vec<MetalGdnLayerState>,
}

impl MetalGdnState {
    pub(crate) fn new(device: &Device, cfg: &Qwen35Config, allocated_layers: usize) -> Self {
        let geometry = MetalGdnStateGeometry {
            architectural_layers: cfg.num_linear_attention_layers(),
            active_layers: cfg.num_active_linear_attention_layers(),
            allocated_layers,
            qkv_dim: cfg.linear_qkv_dim(),
            conv_history: cfg.linear_conv_kernel_dim.saturating_sub(1),
            value_heads: cfg.linear_num_value_heads(),
            key_dim: cfg.linear_key_head_dim,
            value_dim: cfg.linear_value_head_dim,
        };
        debug_assert!(geometry.active_layers <= geometry.architectural_layers);
        debug_assert!(geometry.allocated_layers <= geometry.architectural_layers);

        let conv_buffers: Vec<Buffer> = (0..allocated_layers)
            .map(|i| {
                make_zero_buffer(
                    device,
                    geometry.conv_elements_per_layer(),
                    &format!("gdn_conv_{i}"),
                )
            })
            .collect();
        let s_matrices: Vec<Buffer> = (0..allocated_layers)
            .map(|i| {
                make_zero_buffer(
                    device,
                    geometry.matrix_elements_per_layer(),
                    &format!("gdn_s_{i}"),
                )
            })
            .collect();
        let layers = conv_buffers
            .into_iter()
            .zip(s_matrices)
            .map(|(conv_buffer, s_matrix)| MetalGdnLayerState {
                conv_buffer,
                s_matrix,
            })
            .collect();

        Self {
            geometry,
            precision: MetalGdnStatePrecision::F32,
            layers,
        }
    }

    pub(crate) fn geometry(&self) -> MetalGdnStateGeometry {
        self.geometry
    }

    pub(crate) fn precision(&self) -> MetalGdnStatePrecision {
        self.precision
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, index: usize) -> &MetalGdnLayerState {
        &self.layers[index]
    }

    pub(crate) fn layers(&self) -> std::slice::Iter<'_, MetalGdnLayerState> {
        self.layers.iter()
    }
}
