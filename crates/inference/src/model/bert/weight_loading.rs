//! BERT encoder and classifier weight assembly from safetensors.

use crate::error::InferenceError;
use crate::weights::{
    BertWeights, CrossEncoderWeights, SafetensorsFile, Tensor1D, Tensor2D, TransformerLayerWeights,
};
use std::borrow::Cow;
use std::collections::BTreeSet;

/// **Unstable**: assemble borrowed BERT encoder weights from safetensors.
///
/// Encoder tensors may be flat or share one dot-delimited namespace. Mixed
/// encoder namespaces are rejected before loading tensor data. Views borrow
/// `source`; absent pooler tensors retain their empty-view representation.
///
/// A borrowed result keeps its source lifetime:
///
/// ```rust,no_run
/// use lattice_inference::model::bert::weight_loading::load_weights;
/// use lattice_inference::weights::{BertWeights, SafetensorsFile};
///
/// fn borrowed(source: &SafetensorsFile) -> BertWeights<'_> {
///     load_weights(source, 2, 4).unwrap()
/// }
/// ```
///
/// An owned source cannot lend an escaping static result:
///
/// ```compile_fail
/// use lattice_inference::model::bert::weight_loading::load_weights;
/// use lattice_inference::weights::{BertWeights, SafetensorsFile};
///
/// fn detached(source: SafetensorsFile) -> BertWeights<'static> {
///     load_weights(&source, 2, 4).unwrap()
/// }
/// ```
pub fn load_weights(
    source: &SafetensorsFile,
    num_layers: usize,
    hidden_size: usize,
) -> Result<BertWeights<'_>, InferenceError> {
    let encoder = BertTensorReader::new(source)?;
    let word_shape = encoder.require_shape("embeddings.word_embeddings.weight")?;
    let position_shape = encoder.require_shape("embeddings.position_embeddings.weight")?;
    let token_type_shape = encoder.require_shape("embeddings.token_type_embeddings.weight")?;

    if word_shape.len() != 2 || word_shape[1] != hidden_size {
        return Err(InferenceError::ShapeMismatch {
            name: encoder
                .name("embeddings.word_embeddings.weight")
                .into_owned(),
            expected: vec![word_shape[0], hidden_size],
            actual: word_shape.to_vec(),
        });
    }
    if position_shape.len() != 2 || position_shape[1] != hidden_size {
        return Err(InferenceError::ShapeMismatch {
            name: encoder
                .name("embeddings.position_embeddings.weight")
                .into_owned(),
            expected: vec![position_shape[0], hidden_size],
            actual: position_shape.to_vec(),
        });
    }
    if token_type_shape.len() != 2 || token_type_shape[1] != hidden_size {
        return Err(InferenceError::ShapeMismatch {
            name: encoder
                .name("embeddings.token_type_embeddings.weight")
                .into_owned(),
            expected: vec![token_type_shape[0], hidden_size],
            actual: token_type_shape.to_vec(),
        });
    }

    let word_embeddings = encoder.tensor2d(
        "embeddings.word_embeddings.weight",
        word_shape[0],
        word_shape[1],
    )?;
    let position_embeddings = encoder.tensor2d(
        "embeddings.position_embeddings.weight",
        position_shape[0],
        position_shape[1],
    )?;
    let token_type_embeddings = encoder.tensor2d(
        "embeddings.token_type_embeddings.weight",
        token_type_shape[0],
        token_type_shape[1],
    )?;
    let embedding_layer_norm_weight =
        encoder.tensor1d("embeddings.LayerNorm.weight", hidden_size)?;
    let embedding_layer_norm_bias = encoder.tensor1d("embeddings.LayerNorm.bias", hidden_size)?;

    let intermediate_shape = encoder.require_shape("encoder.layer.0.intermediate.dense.weight")?;
    if intermediate_shape.len() != 2 || intermediate_shape[1] != hidden_size {
        return Err(InferenceError::ShapeMismatch {
            name: encoder
                .name("encoder.layer.0.intermediate.dense.weight")
                .into_owned(),
            expected: vec![intermediate_shape[0], hidden_size],
            actual: intermediate_shape.to_vec(),
        });
    }
    let intermediate_size = intermediate_shape[0];

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("encoder.layer.{i}");
        layers.push(TransformerLayerWeights {
            query_weight: encoder.tensor2d(
                &format!("{prefix}.attention.self.query.weight"),
                hidden_size,
                hidden_size,
            )?,
            query_bias: encoder
                .tensor1d(&format!("{prefix}.attention.self.query.bias"), hidden_size)?,
            key_weight: encoder.tensor2d(
                &format!("{prefix}.attention.self.key.weight"),
                hidden_size,
                hidden_size,
            )?,
            key_bias: encoder
                .tensor1d(&format!("{prefix}.attention.self.key.bias"), hidden_size)?,
            value_weight: encoder.tensor2d(
                &format!("{prefix}.attention.self.value.weight"),
                hidden_size,
                hidden_size,
            )?,
            value_bias: encoder
                .tensor1d(&format!("{prefix}.attention.self.value.bias"), hidden_size)?,
            attn_output_weight: encoder.tensor2d(
                &format!("{prefix}.attention.output.dense.weight"),
                hidden_size,
                hidden_size,
            )?,
            attn_output_bias: encoder.tensor1d(
                &format!("{prefix}.attention.output.dense.bias"),
                hidden_size,
            )?,
            attn_layer_norm_weight: encoder.tensor1d(
                &format!("{prefix}.attention.output.LayerNorm.weight"),
                hidden_size,
            )?,
            attn_layer_norm_bias: encoder.tensor1d(
                &format!("{prefix}.attention.output.LayerNorm.bias"),
                hidden_size,
            )?,
            ffn_intermediate_weight: encoder.tensor2d(
                &format!("{prefix}.intermediate.dense.weight"),
                intermediate_size,
                hidden_size,
            )?,
            ffn_intermediate_bias: encoder.tensor1d(
                &format!("{prefix}.intermediate.dense.bias"),
                intermediate_size,
            )?,
            ffn_output_weight: encoder.tensor2d(
                &format!("{prefix}.output.dense.weight"),
                hidden_size,
                intermediate_size,
            )?,
            ffn_output_bias: encoder
                .tensor1d(&format!("{prefix}.output.dense.bias"), hidden_size)?,
            ffn_layer_norm_weight: encoder
                .tensor1d(&format!("{prefix}.output.LayerNorm.weight"), hidden_size)?,
            ffn_layer_norm_bias: encoder
                .tensor1d(&format!("{prefix}.output.LayerNorm.bias"), hidden_size)?,
        });
    }

    let pooler_weight =
        encoder.tensor2d_optional("pooler.dense.weight", hidden_size, hidden_size)?;
    let pooler_bias = encoder.tensor1d_optional("pooler.dense.bias", hidden_size)?;

    Ok(BertWeights {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        embedding_layer_norm_weight,
        embedding_layer_norm_bias,
        layers,
        pooler_weight,
        pooler_bias,
    })
}

/// **Unstable**: assemble an owned BERT classifier head from safetensors.
///
/// Classifier tensors stay at the top level independently of the encoder
/// namespace. The weight has shape `[1, hidden_size]` or `[hidden_size]`, and
/// the bias has shape `[1]`. Both are copied, so the result does not borrow
/// `source`.
///
/// The owned result can leave a local source behind:
///
/// ```rust,no_run
/// use lattice_inference::model::bert::weight_loading::load_classifier_weights;
/// use lattice_inference::weights::{CrossEncoderWeights, SafetensorsFile};
///
/// fn owned(source: SafetensorsFile) -> CrossEncoderWeights {
///     load_classifier_weights(&source, 4).unwrap()
/// }
/// ```
pub fn load_classifier_weights(
    source: &SafetensorsFile,
    hidden_size: usize,
) -> Result<CrossEncoderWeights, InferenceError> {
    let (weight, weight_shape) = source.get_f32_tensor("classifier.weight")?;
    let classifier_weight = match weight_shape {
        [1, cols] if *cols == hidden_size => weight.to_vec(),
        [cols] if *cols == hidden_size => weight.to_vec(),
        _ => {
            return Err(InferenceError::ShapeMismatch {
                name: "classifier.weight".to_string(),
                expected: vec![1, hidden_size],
                actual: weight_shape.to_vec(),
            });
        }
    };

    let (bias, bias_shape) = source.get_f32_tensor("classifier.bias")?;
    if bias_shape != [1usize] {
        return Err(InferenceError::ShapeMismatch {
            name: "classifier.bias".to_string(),
            expected: vec![1],
            actual: bias_shape.to_vec(),
        });
    }
    let classifier_bias = bias[0];

    Ok(CrossEncoderWeights {
        classifier_weight,
        classifier_bias,
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

struct BertTensorReader<'a> {
    file: &'a SafetensorsFile,
    prefix: &'a str,
}

impl<'a> BertTensorReader<'a> {
    fn new(file: &'a SafetensorsFile) -> Result<Self, InferenceError> {
        let mut prefixes = BTreeSet::new();
        for name in file.tensor_names_iter() {
            // Head shadows cannot establish an encoder namespace.
            if name.ends_with(".classifier.weight") || name.ends_with(".classifier.bias") {
                continue;
            }
            let offset = ["embeddings.", "encoder.layer.", "pooler."]
                .into_iter()
                .flat_map(|family| name.match_indices(family))
                .map(|(offset, _)| offset)
                .filter(|&offset| offset == 0 || name.as_bytes()[offset - 1] == b'.')
                .max();
            if let Some(offset) = offset {
                prefixes.insert(&name[..offset]);
            }
        }
        if prefixes.len() > 1 {
            return Err(InferenceError::InvalidSafetensors(format!(
                "mixed BERT encoder namespaces: {prefixes:?} (empty prefix is unprefixed)"
            )));
        }
        let prefix = prefixes.into_iter().next().unwrap_or("");
        Ok(Self { file, prefix })
    }

    fn name<'n>(&self, name: &'n str) -> Cow<'n, str> {
        if self.prefix.is_empty() {
            Cow::Borrowed(name)
        } else {
            Cow::Owned(format!("{}{name}", self.prefix))
        }
    }

    fn require_shape(&self, name: &str) -> Result<&'a [usize], InferenceError> {
        require_shape(self.file, &self.name(name))
    }

    fn tensor1d(&self, name: &str, len: usize) -> Result<Tensor1D<'a>, InferenceError> {
        self.file.view_1d(&self.name(name), len)
    }

    fn tensor2d(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'a>, InferenceError> {
        self.file.view_2d(&self.name(name), rows, cols)
    }

    fn tensor1d_optional(&self, name: &str, len: usize) -> Result<Tensor1D<'a>, InferenceError> {
        let name = self.name(name);
        if self.file.has_tensor(&name) {
            self.file.view_1d(&name, len)
        } else {
            Ok(Tensor1D { data: &[], len: 0 })
        }
    }

    fn tensor2d_optional(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'a>, InferenceError> {
        let name = self.name(name);
        if self.file.has_tensor(&name) {
            self.file.view_2d(&name, rows, cols)
        } else {
            Ok(Tensor2D {
                data: &[],
                rows: 0,
                cols: 0,
            })
        }
    }
}

#[cfg(test)]
mod namespace_tests;
#[cfg(test)]
mod tests;
