//! A synthetic Inkling 2 configuration with seeded random weights, small enough to run a forward
//! pass in a unit test. This crate's tests use it, and so, under the `test-utils` feature, do the
//! tests of crates that serve the model. It is not a model of the published checkpoint.

use lattice_arch::inkling2_config::{AttentionSpan, Inkling2TextConfig, parse_config_json};

use crate::{
    AttentionWeights, DenseMatrix, ExpertBank, Inkling2Model, Inkling2Weights, KvConvolution,
    LayerWeights, LoopGates, MessageGateWeights, MoeWeights,
};

/// The synthetic `config.json`: 12 layers of width 32, a vocabulary of 37 ids padded to 40, and
/// 64 positions. `PASSES` stands for the loop's pass count, which [`config`] fills in.
pub const CONFIG: &str = r#"{
  "architectures": ["Inkling2LoopForConditionalGeneration"],
  "model_type": "inkling2_loop",
  "eos_token_id": 1,
  "image_token_id": 2,
  "text_config": {
    "max_position_embeddings": 64,
    "hidden_size": 32,
    "num_hidden_layers": 12,
    "vocab_size": 40,
    "unpadded_vocab_size": 37,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 6,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 4,
    "sliding_window_size": 4,
    "d_rel": 4,
    "rel_extent": 4,
    "log_scaling_n_floor": 2,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "qk_l2_clip_threshold": 2.0,
    "rel_l2_clip_threshold": 1.5,
    "sconv_kernel_size": 4,
    "final_logit_softcapping": 30.0,
    "n_layers_per_attn_res_block": 2,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "loop_config": {"core_start": 4, "core_end": 8, "num_passes": PASSES, "gate_rank": 8}
  }
}"#;

/// The synthetic configuration with `passes` loop passes.
pub fn config(passes: usize) -> Inkling2TextConfig {
    parse_config_json(&CONFIG.replace("PASSES", &passes.to_string()))
        .expect("synthetic config parses")
        .text
}

pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    pub(crate) fn unit(&mut self) -> f32 {
        self.next() as f32 / 2_147_483_648.0 * 2.0 - 1.0
    }

    pub(crate) fn byte(&mut self) -> u8 {
        (self.next() >> 8) as u8
    }

    pub(crate) fn values(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.unit() * scale).collect()
    }

    pub(crate) fn around_one(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| 1.0 + 0.2 * self.unit()).collect()
    }

    pub(crate) fn dense(&mut self, rows: usize, cols: usize, scale: f32) -> DenseMatrix {
        DenseMatrix::from_f32(rows, cols, self.values(rows * cols, scale)).expect("f32 matrix")
    }

    pub(crate) fn dense_bf16(&mut self, rows: usize, cols: usize, scale: f32) -> DenseMatrix {
        let bits = self
            .values(rows * cols, scale)
            .into_iter()
            .map(|value| (value.to_bits() >> 16) as u16)
            .collect();
        DenseMatrix::from_bf16_bits(rows, cols, bits).expect("bf16 matrix")
    }

    pub(crate) fn nvfp4_parts(
        &mut self,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        let elements = experts * rows * cols;
        let codes = (0..elements / 2).map(|_| self.byte()).collect();
        let scales = (0..elements / 16)
            .map(|_| 0x28 + self.byte() % 16)
            .collect();
        let multipliers = (0..experts).map(|_| 0.03 + 0.01 * self.unit()).collect();
        (codes, scales, multipliers)
    }

    pub(crate) fn nvfp4(&mut self, experts: usize, rows: usize, cols: usize) -> ExpertBank {
        let (codes, scales, multipliers) = self.nvfp4_parts(experts, rows, cols);
        ExpertBank::nvfp4(experts, rows, cols, codes, scales, multipliers).expect("NVFP4 bank")
    }

    pub(crate) fn mxfp8(&mut self, experts: usize, rows: usize, cols: usize) -> ExpertBank {
        let elements = experts * rows * cols;
        let values = (0..elements)
            .map(|_| {
                let byte = self.byte();
                if byte & 0x7f == 0x7f { byte ^ 1 } else { byte }
            })
            .collect();
        let scales = (0..elements / 32).map(|_| 116 + self.byte() % 4).collect();
        ExpertBank::mxfp8(experts, rows, cols, values, scales).expect("MXFP8 bank")
    }
}

fn gate(rng: &mut Rng, hidden: usize, rank: usize, scale: f32) -> MessageGateWeights {
    MessageGateWeights {
        down: rng.dense(rank, 2 * hidden, scale),
        forget: rng.dense(hidden, rank, scale),
        gain: rng.dense(hidden, rank, scale),
    }
}

fn kv_convolution(
    rng: &mut Rng,
    global: bool,
    channels: usize,
    kv_heads: usize,
    kernel: usize,
    hidden: usize,
) -> KvConvolution {
    if global {
        KvConvolution::Dynamic {
            static_taps: rng.values(kernel * channels, 0.3),
            projection: rng.dense(kv_heads * kernel, hidden, 0.1),
        }
    } else {
        KvConvolution::Static {
            taps: rng.values(channels * kernel, 0.3),
        }
    }
}

fn layer_weights(
    rng: &mut Rng,
    c: &Inkling2TextConfig,
    layer: usize,
    gate_scale: f32,
) -> LayerWeights {
    let h = c.hidden_size;
    let kernel = c.conv_kernel_size;
    let span = c.layer_spans[layer];
    let global = span == AttentionSpan::Global;
    let heads = c.heads(span);
    let q = heads.num_attention_heads * heads.head_dim;
    let kv = heads.num_key_value_heads * heads.head_dim;
    let r = heads.num_attention_heads * c.d_rel;
    let queries = if layer == 0 {
        0
    } else if global {
        3
    } else {
        1
    };
    let norms = if global { 3 } else { 1 };
    let experts = c.n_routed_experts;
    let intermediate = c.moe_intermediate_size;

    let attention_input_queries = (0..queries).map(|_| rng.values(h, 1.0)).collect();
    let mlp_input_query = rng.values(h, 1.0);
    let attention_norms = (0..norms).map(|_| rng.around_one(h)).collect();
    let mlp_norm = rng.around_one(h);
    let attention = AttentionWeights {
        query: rng.dense_bf16(q, h, 0.3),
        query_scale: rng.around_one(q),
        key: rng.dense(kv, h, 0.3),
        value: rng.dense_bf16(kv, h, 0.3),
        key_convolution: kv_convolution(rng, global, kv, heads.num_key_value_heads, kernel, h),
        value_convolution: kv_convolution(rng, global, kv, heads.num_key_value_heads, kernel, h),
        output_gate: rng.dense(q, h, 0.3),
        output: rng.dense(h, q, 0.3),
        sink: rng.dense(heads.num_attention_heads, h, 0.3),
        relative: rng.dense(r, h, 0.3),
        relative_scale: rng.around_one(r),
        relative_profiles: rng.dense(c.d_rel, c.relative_extent(span), 0.5),
    };
    let post_attention_convolution = rng.values(h * kernel, 0.3);
    let post_mlp_convolution = rng.values(h * kernel, 0.3);
    let (gate_up, down) = if layer < 10 {
        (
            rng.nvfp4(experts, 2 * intermediate, h),
            rng.nvfp4(experts, h, intermediate),
        )
    } else {
        (
            rng.mxfp8(experts, 2 * intermediate, h),
            rng.mxfp8(experts, h, intermediate),
        )
    };
    let moe = MoeWeights {
        router: rng.dense(experts, h, 0.5),
        selection_bias: rng.values(experts, 0.1),
        gate_up,
        down,
    };
    let lc = c.loop_config;
    let gates = if (lc.core_start..lc.core_end).contains(&layer) {
        Some(LoopGates {
            query: gate(rng, h, lc.gate_rank, gate_scale),
            key: gate(rng, h, lc.gate_rank, gate_scale),
            value: gate(rng, h, lc.gate_rank, gate_scale),
            mlp: gate(rng, h, lc.gate_rank, gate_scale),
        })
    } else {
        None
    };
    LayerWeights {
        attention_input_queries,
        mlp_input_query,
        attention_norms,
        mlp_norm,
        attention,
        post_attention_convolution,
        post_mlp_convolution,
        moe,
        gates,
    }
}

pub(crate) fn weights(c: &Inkling2TextConfig, seed: u64, gate_scale: f32) -> Inkling2Weights {
    let mut rng = Rng(seed);
    let h = c.hidden_size;
    let core = c.loop_config.core_end - c.loop_config.core_start;
    let embedding = rng.dense_bf16(c.vocab_size, h, 1.0);
    let embedding_norm = rng.around_one(h);
    let layers = (0..c.num_hidden_layers)
        .map(|layer| layer_weights(&mut rng, c, layer, gate_scale))
        .collect();
    let message_queries = (0..4 * core).map(|_| rng.values(h, 1.0)).collect();
    Inkling2Weights {
        embedding,
        embedding_norm,
        layers,
        message_queries,
        output_query: rng.values(h, 1.0),
        final_norm: rng.around_one(h),
        unembedding: rng.dense(c.vocab_size, h, 0.5),
    }
}

/// The synthetic model with `passes` loop passes and loop gates scaled by `gate_scale`, its
/// weights drawn from a fixed seed.
pub fn model(passes: usize, gate_scale: f32) -> Inkling2Model {
    let c = config(passes);
    let w = weights(&c, 0x1234_5678, gate_scale);
    Inkling2Model::new(c, w).expect("synthetic model builds")
}
