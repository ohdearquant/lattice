//! Qwen3.5 checkpoint tensor names: the per-layer prefix and the required-tensor list a config implies.
use crate::qwen35_config::Qwen35Config;
use lattice_inference_core::InferenceError;

/// Nested Qwen3.5 text-model tensor namespace.
#[doc(hidden)]
pub const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";
/// Flat Qwen3.5 text-model tensor namespace.
#[doc(hidden)]
pub const FLAT_MODEL_PREFIX: &str = "model.";
/// Embedding tensor that identifies the nested text-model namespace.
#[doc(hidden)]
pub const LANGUAGE_MODEL_EMBED: &str = "model.language_model.embed_tokens.weight";
/// Embedding tensor that identifies the flat text-model namespace.
#[doc(hidden)]
pub const FLAT_MODEL_EMBED: &str = "model.embed_tokens.weight";

/// Build a native Q4 tensor filename, preserving Unicode alphanumeric characters and hyphens.
/// Other tensor-name characters become underscores; `ext` is appended verbatim after a dot.
#[doc(hidden)]
pub fn q4_tensor_filename(tensor_name: &str, ext: &str) -> String {
    let sanitized: String = tensor_name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{sanitized}.{ext}")
}

/// Inspect a native Q4 embedding anchor without following its final component.
/// Returns the inspected regular file's metadata, or `None` when the anchor is absent.
///
/// # Errors
/// Returns [`std::io::ErrorKind::InvalidData`] for symlinks and non-regular files.
/// Other inspection errors retain their original kind and contextual message.
#[doc(hidden)]
pub fn inspect_q4_embedding_anchor(
    path: &std::path::Path,
) -> std::io::Result<Option<std::fs::Metadata>> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Q4 embedding anchor {}: final component is a symlink",
                path.display()
            ),
        )),
        Ok(metadata) if metadata.is_file() => Ok(Some(metadata)),
        Ok(_) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Q4 embedding anchor {} is not a regular file",
                path.display()
            ),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(std::io::Error::new(
            error.kind(),
            format!("inspect {}: {error}", path.display()),
        )),
    }
}

/// Resolve a checkpoint's text-model namespace from its embedding anchors.
///
/// Exactly one embedding anchor must identify the namespace.
///
/// # Errors
/// Returns an invalid-safetensors error when both anchors are present, or a
/// missing-tensor error naming the canonical embedding when neither is present.
#[doc(hidden)]
pub fn resolve_tensor_prefix(
    has_language_model: bool,
    has_flat_model: bool,
) -> Result<&'static str, InferenceError> {
    match (has_language_model, has_flat_model) {
        (true, true) => Err(InferenceError::InvalidSafetensors(format!(
            "checkpoint contains both {LANGUAGE_MODEL_EMBED} and {FLAT_MODEL_EMBED}"
        ))),
        (true, false) => Ok(LANGUAGE_MODEL_PREFIX),
        (false, true) => Ok(FLAT_MODEL_PREFIX),
        (false, false) => Err(InferenceError::MissingTensor(
            LANGUAGE_MODEL_EMBED.to_owned(),
        )),
    }
}

/// Resolve a text namespace for inventory diagnostics when embedding anchors may be absent.
///
/// Exact embedding anchors take priority. Without either anchor, hints are `norm.weight`
/// and the `layers.` prefix under each text-namespace prefix constant. Flat-only hints
/// select the flat namespace; mixed or absent hints fall back to the canonical namespace.
///
/// # Errors
/// Preserves [`resolve_tensor_prefix`]'s ambiguity error when both embedding anchors exist.
#[doc(hidden)]
pub fn resolve_inventory_tensor_prefix<'a>(
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<&'static str, InferenceError> {
    let mut has_language_model = false;
    let mut has_flat_model = false;
    let mut has_language_model_hint = false;
    let mut has_flat_hint = false;
    let is_hint = |name: &str, prefix: &str| {
        name.strip_prefix(prefix)
            .is_some_and(|suffix| suffix == "norm.weight" || suffix.starts_with("layers."))
    };
    for name in tensor_names {
        has_language_model |= name == LANGUAGE_MODEL_EMBED;
        has_flat_model |= name == FLAT_MODEL_EMBED;
        has_language_model_hint |= is_hint(name, LANGUAGE_MODEL_PREFIX);
        has_flat_hint |= is_hint(name, FLAT_MODEL_PREFIX);
    }
    if has_language_model || has_flat_model {
        return resolve_tensor_prefix(has_language_model, has_flat_model);
    }
    if has_flat_hint && !has_language_model_hint {
        Ok(FLAT_MODEL_PREFIX)
    } else {
        Ok(LANGUAGE_MODEL_PREFIX)
    }
}

/// Canonical per-layer tensor-name prefix for a Qwen3.5 checkpoint
/// (`model.language_model.layers.{idx}`). Canonical checkpoint consumers use
/// this function, while loaders use [`qwen_layer_tensor_prefix_with_prefix`]
/// after resolving the namespace. Both share the same name construction so
/// the required-tensor list, QuaRot converter and artifact validator stay in sync.
pub fn qwen_layer_tensor_prefix(idx: usize) -> String {
    qwen_layer_tensor_prefix_with_prefix(LANGUAGE_MODEL_PREFIX, idx)
}

/// Build a layer prefix using the namespace returned by [`resolve_tensor_prefix`].
#[doc(hidden)]
pub fn qwen_layer_tensor_prefix_with_prefix(tensor_prefix: &str, idx: usize) -> String {
    format!("{tensor_prefix}layers.{idx}")
}

/// Return the complete list of required tensor names for a given config.
pub fn qwen_required_tensor_names(cfg: &Qwen35Config) -> Vec<String> {
    qwen_required_tensor_names_with_prefix(cfg, LANGUAGE_MODEL_PREFIX)
}

/// Build required names using the namespace returned by [`resolve_tensor_prefix`].
#[doc(hidden)]
pub fn qwen_required_tensor_names_with_prefix(
    cfg: &Qwen35Config,
    tensor_prefix: &str,
) -> Vec<String> {
    let mut names = Vec::new();
    names.push(format!("{tensor_prefix}embed_tokens.weight"));
    names.push(format!("{tensor_prefix}norm.weight"));
    if !cfg.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }

    for i in 0..cfg.num_hidden_layers {
        let prefix = qwen_layer_tensor_prefix_with_prefix(tensor_prefix, i);
        names.push(format!("{prefix}.input_layernorm.weight"));
        names.push(format!("{prefix}.post_attention_layernorm.weight"));

        if cfg.is_full_attention(i) {
            names.push(format!("{prefix}.self_attn.q_proj.weight"));
            names.push(format!("{prefix}.self_attn.k_proj.weight"));
            names.push(format!("{prefix}.self_attn.v_proj.weight"));
            names.push(format!("{prefix}.self_attn.o_proj.weight"));
            names.push(format!("{prefix}.self_attn.q_norm.weight"));
            names.push(format!("{prefix}.self_attn.k_norm.weight"));
        } else {
            names.push(format!("{prefix}.linear_attn.in_proj_qkv.weight"));
            names.push(format!("{prefix}.linear_attn.in_proj_z.weight"));
            names.push(format!("{prefix}.linear_attn.in_proj_b.weight"));
            names.push(format!("{prefix}.linear_attn.in_proj_a.weight"));
            names.push(format!("{prefix}.linear_attn.A_log"));
            names.push(format!("{prefix}.linear_attn.dt_bias"));
            names.push(format!("{prefix}.linear_attn.conv1d.weight"));
            names.push(format!("{prefix}.linear_attn.norm.weight"));
            names.push(format!("{prefix}.linear_attn.out_proj.weight"));
        }

        if cfg.is_moe() {
            names.push(format!("{prefix}.mlp.gate.weight"));
            names.push(format!("{prefix}.mlp.experts.gate_up_proj"));
            names.push(format!("{prefix}.mlp.experts.down_proj"));
            names.push(format!("{prefix}.mlp.shared_expert.gate_proj.weight"));
            names.push(format!("{prefix}.mlp.shared_expert.up_proj.weight"));
            names.push(format!("{prefix}.mlp.shared_expert.down_proj.weight"));
            names.push(format!("{prefix}.mlp.shared_expert_gate.weight"));
        } else {
            names.push(format!("{prefix}.mlp.gate_proj.weight"));
            names.push(format!("{prefix}.mlp.up_proj.weight"));
            names.push(format!("{prefix}.mlp.down_proj.weight"));
        }
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_tensor_filename_preserves_native_naming_rules() {
        for (name, ext, expected) in [
            (
                "model.language_model.embed_tokens.weight",
                "q4",
                "model_language_model_embed_tokens_weight.q4",
            ),
            (
                "model.embed_tokens.weight",
                "q4",
                "model_embed_tokens_weight.q4",
            ),
            ("layer-2.weight_0", "f16", "layer-2_weight_0.f16"),
            ("a-b_c—d", "q4", "a-b_c_d.q4"),
            ("tensor/part: value", "q4", "tensor_part__value.q4"),
            ("é中٣", "q4", "é中٣.q4"),
            ("e\u{301}.😀", "q4", "e___.q4"),
            ("tensor", "", "tensor."),
            ("", "q4", ".q4"),
            ("tensor", ".Q4+raw", "tensor..Q4+raw"),
        ] {
            assert_eq!(q4_tensor_filename(name, ext), expected, "{name:?}, {ext:?}");
        }
    }

    #[test]
    fn q4_embedding_anchor_requires_a_regular_file() {
        for (name, opposite) in [
            (
                "model_language_model_embed_tokens_weight.q4",
                "model_embed_tokens_weight.q4",
            ),
            (
                "model_embed_tokens_weight.q4",
                "model_language_model_embed_tokens_weight.q4",
            ),
        ] {
            for keep_opposite in [true, false] {
                let dir = tempfile::tempdir().unwrap();
                if keep_opposite {
                    std::fs::write(dir.path().join(opposite), b"anchor").unwrap();
                }
                let anchor = dir.path().join(name);
                assert!(inspect_q4_embedding_anchor(&anchor).unwrap().is_none());
                std::fs::write(&anchor, b"anchor").unwrap();
                let metadata = inspect_q4_embedding_anchor(&anchor)
                    .unwrap()
                    .expect("regular anchor metadata");
                assert!(metadata.is_file());
                assert_eq!(metadata.len(), 6);
                std::fs::remove_file(&anchor).unwrap();
                std::fs::create_dir(&anchor).unwrap();
                let error = inspect_q4_embedding_anchor(&anchor).unwrap_err();
                assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
                assert_eq!(
                    error.to_string(),
                    format!(
                        "Q4 embedding anchor {} is not a regular file",
                        anchor.display()
                    ),
                    "nonregular anchor must be refused"
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn q4_embedding_anchor_preserves_inspection_errors() {
        for name in [
            "model_language_model_embed_tokens_weight.q4",
            "model_embed_tokens_weight.q4",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let anchor = dir.path().join(name);
            std::fs::write(&anchor, b"anchor").unwrap();
            assert!(inspect_q4_embedding_anchor(&anchor).unwrap().is_some());
            let loop_dir = dir.path().join("loop");
            std::os::unix::fs::symlink(&loop_dir, &loop_dir).unwrap();
            assert!(
                std::fs::symlink_metadata(&loop_dir)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            let anchor = loop_dir.join(name);
            let source_error = std::fs::symlink_metadata(&anchor)
                .expect_err("intermediate-path metadata error fixture");
            let err = inspect_q4_embedding_anchor(&anchor)
                .expect_err("intermediate-path metadata errors must refuse");
            assert_eq!(err.kind(), source_error.kind());
            assert_eq!(
                err.to_string(),
                format!("inspect {}: {source_error}", anchor.display())
            );
            assert!(
                err.to_string()
                    .starts_with(&format!("inspect {}:", anchor.display())),
                "metadata errors must retain inspection context: {err}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn q4_embedding_anchor_refuses_symlinks() {
        for name in [
            "model_language_model_embed_tokens_weight.q4",
            "model_embed_tokens_weight.q4",
        ] {
            for target_kind in ["regular", "missing", "directory"] {
                let dir = tempfile::tempdir().unwrap();
                let anchor = dir.path().join(name);
                std::fs::write(&anchor, b"anchor").unwrap();
                assert!(inspect_q4_embedding_anchor(&anchor).unwrap().is_some());
                let referent = dir.path().join("embedding-payload");
                if target_kind == "regular" {
                    std::fs::rename(&anchor, &referent).unwrap();
                } else {
                    std::fs::remove_file(&anchor).unwrap();
                    if target_kind == "directory" {
                        std::fs::create_dir(&referent).unwrap();
                    }
                }
                std::os::unix::fs::symlink(&referent, &anchor).unwrap();
                assert!(
                    std::fs::symlink_metadata(&anchor)
                        .unwrap()
                        .file_type()
                        .is_symlink()
                );
                match target_kind {
                    "regular" => assert!(referent.is_file()),
                    "directory" => assert!(referent.is_dir()),
                    _ => assert!(!referent.exists()),
                }
                let error = inspect_q4_embedding_anchor(&anchor).unwrap_err();
                assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
                assert_eq!(
                    error.to_string(),
                    format!(
                        "Q4 embedding anchor {}: final component is a symlink",
                        anchor.display()
                    ),
                    "final-component symlinks must refuse: {name} {target_kind}"
                );
            }
        }
    }

    #[test]
    fn inventory_namespace_uses_missing_anchor_hints() {
        let cases: [(&str, &[&str], &str); 8] = [
            (
                "flat layer",
                &["model.layers.0.self_attn.q_proj.weight"],
                "model.",
            ),
            ("flat norm", &["model.norm.weight"], "model."),
            (
                "canonical layer",
                &["model.language_model.layers.0.self_attn.q_proj.weight"],
                "model.language_model.",
            ),
            (
                "canonical norm",
                &["model.language_model.norm.weight"],
                "model.language_model.",
            ),
            ("empty", &[], "model.language_model."),
            (
                "unrelated towers",
                &[
                    "lm_head.weight",
                    "mtp.layers.0.weight",
                    "model.vision_tower.weight",
                    "model.audio_tower.weight",
                ],
                "model.language_model.",
            ),
            (
                "mixed text hints",
                &[
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.language_model.norm.weight",
                ],
                "model.language_model.",
            ),
            (
                "flat with unrelated canonical-prefix names",
                &[
                    "model.norm.weight",
                    "model.language_model.vision_tower.weight",
                    "model.language_model.mtp.layers.0.weight",
                    "model.language_model.lm_head.weight",
                    "model.language_model.layers_extra.0.weight",
                ],
                "model.",
            ),
        ];
        for (label, names, expected) in cases {
            assert_eq!(
                resolve_inventory_tensor_prefix(names.iter().copied()).unwrap(),
                expected,
                "{label}: forward order"
            );
            assert_eq!(
                resolve_inventory_tensor_prefix(names.iter().rev().copied()).unwrap(),
                expected,
                "{label}: reverse order"
            );
        }
    }

    #[test]
    fn inventory_namespace_preserves_anchor_priority_and_ambiguity() {
        for (names, expected) in [
            (
                [
                    "model.language_model.embed_tokens.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.norm.weight",
                ],
                "model.language_model.",
            ),
            (
                [
                    "model.embed_tokens.weight",
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    "model.language_model.norm.weight",
                ],
                "model.",
            ),
        ] {
            assert_eq!(resolve_inventory_tensor_prefix(names).unwrap(), expected);
            assert_eq!(
                resolve_inventory_tensor_prefix(names.into_iter().rev()).unwrap(),
                expected
            );
        }
        let names = [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ];
        for result in [
            resolve_inventory_tensor_prefix(names),
            resolve_inventory_tensor_prefix(names.into_iter().rev()),
        ] {
            assert!(matches!(result,
                Err(InferenceError::InvalidSafetensors(message))
                    if message == "checkpoint contains both model.language_model.embed_tokens.weight and model.embed_tokens.weight"));
        }
    }

    #[test]
    fn canonical_embedding_selects_canonical_namespace() {
        assert_eq!(
            resolve_tensor_prefix(true, false).unwrap(),
            "model.language_model."
        );
        assert_eq!(
            LANGUAGE_MODEL_EMBED,
            "model.language_model.embed_tokens.weight"
        );
    }

    #[test]
    fn flat_embedding_selects_flat_namespace() {
        assert_eq!(resolve_tensor_prefix(false, true).unwrap(), "model.");
        assert_eq!(FLAT_MODEL_EMBED, "model.embed_tokens.weight");
    }

    #[test]
    fn both_embedding_anchors_are_ambiguous() {
        assert!(matches!(resolve_tensor_prefix(true, true),
            Err(InferenceError::InvalidSafetensors(message))
                if message.contains(LANGUAGE_MODEL_EMBED) && message.contains(FLAT_MODEL_EMBED)));
    }

    #[test]
    fn neither_embedding_anchor_reports_canonical_missing_tensor() {
        assert!(
            matches!(resolve_tensor_prefix(false, false),
            Err(InferenceError::MissingTensor(name)) if name == "model.language_model.embed_tokens.weight"),
            "a checkpoint without either embedding anchor must refuse namespace selection"
        );
    }
}
