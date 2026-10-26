//! Qwen3 weight assembly over neutral safetensors reads.

use crate::error::InferenceError;
use crate::weights::{
    QwenLayerWeights, QwenWeights, SafetensorsFile, ShardedQwenBacking, ShardedSafetensors,
    Tensor1D, Tensor2D, TensorSource,
};

/// **Unstable**: assemble Qwen3 weights borrowing from a single safetensors file.
///
/// Raw tensor views borrow `source`; fused QKV and gate/up buffers are owned by
/// the result. Tensor names and validation follow the Qwen3 checkpoint recipe.
///
/// ```
/// use lattice_inference::model::qwen::weight_loading::load_weights;
/// use lattice_inference::weights::{QwenWeights, SafetensorsFile};
/// fn borrow(source: &SafetensorsFile) -> QwenWeights<'_> {
///     load_weights(source, 0, 2, 2, 1, 1, 3).unwrap()
/// }
/// ```
///
/// ```compile_fail
/// use lattice_inference::model::qwen::weight_loading::load_weights;
/// use lattice_inference::weights::{QwenWeights, SafetensorsFile};
/// fn escape(source: SafetensorsFile) -> QwenWeights<'static> {
///     load_weights(&source, 0, 2, 2, 1, 1, 3).unwrap()
/// }
/// ```
pub fn load_weights(
    source: &SafetensorsFile,
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
) -> Result<QwenWeights<'_>, InferenceError> {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let embed_shape = require_shape(source, "embed_tokens.weight")?;
    if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
        return Err(InferenceError::ShapeMismatch {
            name: "embed_tokens.weight".into(),
            expected: vec![embed_shape[0], hidden_size],
            actual: embed_shape.to_vec(),
        });
    }

    let embed_tokens = source.view_2d("embed_tokens.weight", embed_shape[0], hidden_size)?;
    let norm_weight = source.view_1d("norm.weight", hidden_size)?;

    let qkv_out_dim = q_dim + 2 * kv_dim;
    let gate_up_out_dim = 2 * intermediate_size;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let p = format!("layers.{i}");

        // Load individual weight tensors first.
        let q_proj_weight =
            source.view_2d(&format!("{p}.self_attn.q_proj.weight"), q_dim, hidden_size)?;
        let k_proj_weight =
            source.view_2d(&format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden_size)?;
        let v_proj_weight =
            source.view_2d(&format!("{p}.self_attn.v_proj.weight"), kv_dim, hidden_size)?;
        let gate_proj_weight = source.view_2d(
            &format!("{p}.mlp.gate_proj.weight"),
            intermediate_size,
            hidden_size,
        )?;
        let up_proj_weight = source.view_2d(
            &format!("{p}.mlp.up_proj.weight"),
            intermediate_size,
            hidden_size,
        )?;

        // Build fused QKV weight: [q_dim + 2*kv_dim, hidden_size]
        // Rows of W_q, W_k, W_v concatenated vertically.
        let mut fused_qkv = Vec::with_capacity(qkv_out_dim * hidden_size);
        fused_qkv.extend_from_slice(q_proj_weight.data); // [q_dim * hidden_size]
        fused_qkv.extend_from_slice(k_proj_weight.data); // [kv_dim * hidden_size]
        fused_qkv.extend_from_slice(v_proj_weight.data); // [kv_dim * hidden_size]

        // Build fused gate+up weight: [2*intermediate_size, hidden_size]
        // Rows of W_gate, W_up concatenated vertically.
        let mut fused_gate_up = Vec::with_capacity(gate_up_out_dim * hidden_size);
        fused_gate_up.extend_from_slice(gate_proj_weight.data); // [intermediate * hidden]
        fused_gate_up.extend_from_slice(up_proj_weight.data); // [intermediate * hidden]

        layers.push(QwenLayerWeights {
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight: source.view_2d(
                &format!("{p}.self_attn.o_proj.weight"),
                hidden_size,
                q_dim,
            )?,
            q_norm_weight: source.view_1d(&format!("{p}.self_attn.q_norm.weight"), head_dim)?,
            k_norm_weight: source.view_1d(&format!("{p}.self_attn.k_norm.weight"), head_dim)?,
            input_layernorm_weight: source
                .view_1d(&format!("{p}.input_layernorm.weight"), hidden_size)?,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight: source.view_2d(
                &format!("{p}.mlp.down_proj.weight"),
                hidden_size,
                intermediate_size,
            )?,
            post_attention_layernorm_weight: source
                .view_1d(&format!("{p}.post_attention_layernorm.weight"), hidden_size)?,
            fused_qkv,
            qkv_out_dim,
            fused_gate_up,
            gate_up_out_dim,
        });
    }

    Ok(QwenWeights {
        embed_tokens,
        norm_weight,
        layers,
    })
}

/// **Unstable**: owned raw Qwen3 tensors with receiver-borrowed assembly.
///
/// The owner retains every raw tensor allocation independently of the source
/// checkpoint. Its fields are private; assembled views cannot outlive it.
#[derive(Debug)]
pub struct OwnedQwenWeights {
    backing: ShardedQwenBacking,
    num_layers: usize,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    intermediate_size: usize,
    qkv_out_dim: usize,
    gate_up_out_dim: usize,
    embed_vocab: usize,
}

impl OwnedQwenWeights {
    /// **Unstable**: borrow raw tensor views and allocate fresh fused buffers.
    ///
    /// Each call concatenates Q/K/V and gate/up into new owned vectors. The
    /// model calls this once during construction, before its Metal attempt.
    /// The raw views borrow `self`, independently of any tensor-name string.
    ///
    /// ```
    /// use lattice_inference::model::qwen::weight_loading::OwnedQwenWeights;
    /// use lattice_inference::weights::QwenWeights;
    /// fn borrow(owner: &OwnedQwenWeights) -> QwenWeights<'_> {
    ///     owner.assemble_borrowed_weights()
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use lattice_inference::model::qwen::weight_loading::OwnedQwenWeights;
    /// use lattice_inference::weights::QwenWeights;
    /// fn escape(owner: OwnedQwenWeights) -> QwenWeights<'static> {
    ///     owner.assemble_borrowed_weights()
    /// }
    /// ```
    ///
    /// ```
    /// use lattice_inference::model::qwen::weight_loading::OwnedQwenWeights;
    /// fn move_after_borrow(owner: OwnedQwenWeights) -> OwnedQwenWeights {
    ///     let weights = owner.assemble_borrowed_weights();
    ///     assert_eq!(weights.norm_weight.len, weights.norm_weight.data.len());
    ///     drop(weights);
    ///     owner
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use lattice_inference::model::qwen::weight_loading::OwnedQwenWeights;
    /// fn move_during_borrow(owner: OwnedQwenWeights) -> OwnedQwenWeights {
    ///     let weights = owner.assemble_borrowed_weights();
    ///     let moved = owner;
    ///     assert_eq!(weights.norm_weight.len, weights.norm_weight.data.len());
    ///     moved
    /// }
    /// ```
    ///
    /// ```
    /// use lattice_inference::model::qwen::weight_loading::OwnedQwenWeights;
    /// fn drop_after_borrow(owner: OwnedQwenWeights) {
    ///     let weights = owner.assemble_borrowed_weights();
    ///     assert_eq!(weights.norm_weight.len, weights.norm_weight.data.len());
    ///     drop(weights);
    ///     drop(owner);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use lattice_inference::model::qwen::weight_loading::OwnedQwenWeights;
    /// fn drop_during_borrow(owner: OwnedQwenWeights) {
    ///     let weights = owner.assemble_borrowed_weights();
    ///     drop(owner);
    ///     assert_eq!(weights.norm_weight.len, weights.norm_weight.data.len());
    /// }
    /// ```
    pub fn assemble_borrowed_weights(&self) -> QwenWeights<'_> {
        let b = &self.backing;
        let Self {
            num_layers,
            hidden_size,
            q_dim,
            kv_dim,
            head_dim,
            intermediate_size,
            qkv_out_dim,
            gate_up_out_dim,
            embed_vocab,
            ..
        } = *self;
        let embed_tokens = Tensor2D {
            data: &b.embed_tokens,
            rows: embed_vocab,
            cols: hidden_size,
        };
        let norm_weight = Tensor1D {
            data: &b.norm_weight,
            len: hidden_size,
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // Build fused QKV: [q_dim + 2*kv_dim, hidden_size]
            let mut fused_qkv = Vec::with_capacity(qkv_out_dim * hidden_size);
            fused_qkv.extend_from_slice(&b.q_proj[i]);
            fused_qkv.extend_from_slice(&b.k_proj[i]);
            fused_qkv.extend_from_slice(&b.v_proj[i]);

            // Build fused gate+up: [2*intermediate_size, hidden_size]
            let mut fused_gate_up = Vec::with_capacity(gate_up_out_dim * hidden_size);
            fused_gate_up.extend_from_slice(&b.gate_proj[i]);
            fused_gate_up.extend_from_slice(&b.up_proj[i]);

            layers.push(QwenLayerWeights {
                q_proj_weight: Tensor2D {
                    data: &b.q_proj[i],
                    rows: q_dim,
                    cols: hidden_size,
                },
                k_proj_weight: Tensor2D {
                    data: &b.k_proj[i],
                    rows: kv_dim,
                    cols: hidden_size,
                },
                v_proj_weight: Tensor2D {
                    data: &b.v_proj[i],
                    rows: kv_dim,
                    cols: hidden_size,
                },
                o_proj_weight: Tensor2D {
                    data: &b.o_proj[i],
                    rows: hidden_size,
                    cols: q_dim,
                },
                q_norm_weight: Tensor1D {
                    data: &b.q_norm[i],
                    len: head_dim,
                },
                k_norm_weight: Tensor1D {
                    data: &b.k_norm[i],
                    len: head_dim,
                },
                input_layernorm_weight: Tensor1D {
                    data: &b.input_ln[i],
                    len: hidden_size,
                },
                gate_proj_weight: Tensor2D {
                    data: &b.gate_proj[i],
                    rows: intermediate_size,
                    cols: hidden_size,
                },
                up_proj_weight: Tensor2D {
                    data: &b.up_proj[i],
                    rows: intermediate_size,
                    cols: hidden_size,
                },
                down_proj_weight: Tensor2D {
                    data: &b.down_proj[i],
                    rows: hidden_size,
                    cols: intermediate_size,
                },
                post_attention_layernorm_weight: Tensor1D {
                    data: &b.post_ln[i],
                    len: hidden_size,
                },
                fused_qkv,
                qkv_out_dim,
                fused_gate_up,
                gate_up_out_dim,
            });
        }

        QwenWeights {
            embed_tokens,
            norm_weight,
            layers,
        }
    }
}

/// **Unstable**: read a sharded Qwen3 checkpoint into an independent owner.
///
/// Converted allocations are transferred by the concrete owned tensor getter;
/// F32 views without a cached conversion are copied. No views escape this call.
/// Each tensor is checked before reading the next tensor in the Qwen3 recipe.
///
/// ```
/// use lattice_inference::model::qwen::weight_loading::{load_owned_weights, OwnedQwenWeights};
/// use lattice_inference::weights::ShardedSafetensors;
/// fn load(source: &mut ShardedSafetensors) -> OwnedQwenWeights {
///     load_owned_weights(source, 0, 2, 2, 1, 1, 3).unwrap()
/// }
/// ```
pub fn load_owned_weights(
    source: &mut ShardedSafetensors,
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
) -> Result<OwnedQwenWeights, InferenceError> {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = q_dim + 2 * kv_dim;
    let gate_up_out_dim = 2 * intermediate_size;

    // --- Load global tensors ---
    let (embed_data, embed_shape) = source.get_f32_tensor_owned("embed_tokens.weight")?;
    if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
        return Err(InferenceError::ShapeMismatch {
            name: "embed_tokens.weight".into(),
            expected: vec![embed_shape[0], hidden_size],
            actual: embed_shape,
        });
    }
    let embed_vocab = embed_shape[0];

    let (norm_data, norm_shape) = source.get_f32_tensor_owned("norm.weight")?;
    if norm_shape != [hidden_size] {
        return Err(InferenceError::ShapeMismatch {
            name: "norm.weight".into(),
            expected: vec![hidden_size],
            actual: norm_shape,
        });
    }

    // --- Load per-layer tensors ---
    let mut q_proj_vecs = Vec::with_capacity(num_layers);
    let mut k_proj_vecs = Vec::with_capacity(num_layers);
    let mut v_proj_vecs = Vec::with_capacity(num_layers);
    let mut o_proj_vecs = Vec::with_capacity(num_layers);
    let mut q_norm_vecs = Vec::with_capacity(num_layers);
    let mut k_norm_vecs = Vec::with_capacity(num_layers);
    let mut input_ln_vecs = Vec::with_capacity(num_layers);
    let mut gate_proj_vecs = Vec::with_capacity(num_layers);
    let mut up_proj_vecs = Vec::with_capacity(num_layers);
    let mut down_proj_vecs = Vec::with_capacity(num_layers);
    let mut post_ln_vecs = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let p = format!("layers.{i}");

        let (q, qs) = source.get_f32_tensor_owned(&format!("{p}.self_attn.q_proj.weight"))?;
        if qs != [q_dim, hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.self_attn.q_proj.weight"),
                expected: vec![q_dim, hidden_size],
                actual: qs,
            });
        }
        q_proj_vecs.push(q);

        let (k, ks) = source.get_f32_tensor_owned(&format!("{p}.self_attn.k_proj.weight"))?;
        if ks != [kv_dim, hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.self_attn.k_proj.weight"),
                expected: vec![kv_dim, hidden_size],
                actual: ks,
            });
        }
        k_proj_vecs.push(k);

        let (v, vs) = source.get_f32_tensor_owned(&format!("{p}.self_attn.v_proj.weight"))?;
        if vs != [kv_dim, hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.self_attn.v_proj.weight"),
                expected: vec![kv_dim, hidden_size],
                actual: vs,
            });
        }
        v_proj_vecs.push(v);

        let (o, os) = source.get_f32_tensor_owned(&format!("{p}.self_attn.o_proj.weight"))?;
        if os != [hidden_size, q_dim] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.self_attn.o_proj.weight"),
                expected: vec![hidden_size, q_dim],
                actual: os,
            });
        }
        o_proj_vecs.push(o);

        let (qn, qns) = source.get_f32_tensor_owned(&format!("{p}.self_attn.q_norm.weight"))?;
        if qns != [head_dim] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.self_attn.q_norm.weight"),
                expected: vec![head_dim],
                actual: qns,
            });
        }
        q_norm_vecs.push(qn);

        let (kn, kns) = source.get_f32_tensor_owned(&format!("{p}.self_attn.k_norm.weight"))?;
        if kns != [head_dim] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.self_attn.k_norm.weight"),
                expected: vec![head_dim],
                actual: kns,
            });
        }
        k_norm_vecs.push(kn);

        let (iln, ilns) = source.get_f32_tensor_owned(&format!("{p}.input_layernorm.weight"))?;
        if ilns != [hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.input_layernorm.weight"),
                expected: vec![hidden_size],
                actual: ilns,
            });
        }
        input_ln_vecs.push(iln);

        let (g, gs) = source.get_f32_tensor_owned(&format!("{p}.mlp.gate_proj.weight"))?;
        if gs != [intermediate_size, hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.mlp.gate_proj.weight"),
                expected: vec![intermediate_size, hidden_size],
                actual: gs,
            });
        }
        gate_proj_vecs.push(g);

        let (u, us) = source.get_f32_tensor_owned(&format!("{p}.mlp.up_proj.weight"))?;
        if us != [intermediate_size, hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.mlp.up_proj.weight"),
                expected: vec![intermediate_size, hidden_size],
                actual: us,
            });
        }
        up_proj_vecs.push(u);

        let (d, ds) = source.get_f32_tensor_owned(&format!("{p}.mlp.down_proj.weight"))?;
        if ds != [hidden_size, intermediate_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.mlp.down_proj.weight"),
                expected: vec![hidden_size, intermediate_size],
                actual: ds,
            });
        }
        down_proj_vecs.push(d);

        let (pln, plns) =
            source.get_f32_tensor_owned(&format!("{p}.post_attention_layernorm.weight"))?;
        if plns != [hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: format!("{p}.post_attention_layernorm.weight"),
                expected: vec![hidden_size],
                actual: plns,
            });
        }
        post_ln_vecs.push(pln);
    }

    let backing = ShardedQwenBacking {
        embed_tokens: embed_data,
        norm_weight: norm_data,
        q_proj: q_proj_vecs,
        k_proj: k_proj_vecs,
        v_proj: v_proj_vecs,
        o_proj: o_proj_vecs,
        q_norm: q_norm_vecs,
        k_norm: k_norm_vecs,
        input_ln: input_ln_vecs,
        gate_proj: gate_proj_vecs,
        up_proj: up_proj_vecs,
        down_proj: down_proj_vecs,
        post_ln: post_ln_vecs,
    };

    Ok(OwnedQwenWeights {
        backing,
        num_layers,
        hidden_size,
        q_dim,
        kv_dim,
        head_dim,
        intermediate_size,
        qkv_out_dim,
        gate_up_out_dim,
        embed_vocab,
    })
}

fn require_shape<'a>(
    source: &'a SafetensorsFile,
    name: &str,
) -> Result<&'a [usize], InferenceError> {
    source
        .tensor_shape(name)
        .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
}

#[cfg(test)]
mod tests;
