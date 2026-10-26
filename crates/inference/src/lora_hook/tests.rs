use super::*;

#[test]
fn qwen35_projection_shape_covers_full_attention_modules() {
    let config = Qwen35Config::qwen35_0_8b();
    let full_layer = 3;
    let cases = [
        (
            "q_proj",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 4096,
            },
        ),
        (
            "k_proj",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 512,
            },
        ),
        (
            "v_proj",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 512,
            },
        ),
        (
            "o_proj",
            LoraProjectionShape {
                d_in: 2048,
                d_out: 1024,
            },
        ),
    ];

    for (module, expected) in cases {
        assert_eq!(
            qwen35_projection_shape(&config, full_layer, module),
            Ok(expected),
            "unexpected shape for {module}"
        );
    }
}

#[test]
fn qwen35_projection_shape_covers_gdn_modules_including_alpha_and_beta() {
    let config = Qwen35Config::qwen35_0_8b();
    let gdn_layer = 0;
    let cases = [
        (
            "in_proj_qkv",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 6144,
            },
        ),
        (
            "in_proj_z",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 2048,
            },
        ),
        (
            "in_proj_b",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 16,
            },
        ),
        (
            "in_proj_a",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 16,
            },
        ),
        (
            "out_proj",
            LoraProjectionShape {
                d_in: 2048,
                d_out: 1024,
            },
        ),
    ];

    for (module, expected) in cases {
        assert_eq!(
            qwen35_projection_shape(&config, gdn_layer, module),
            Ok(expected),
            "unexpected shape for {module}"
        );
    }
}

#[test]
fn qwen35_projection_shape_uses_value_heads_for_gdn_alpha_and_beta() {
    let config = Qwen35Config::qwen36_35b_a3b();
    assert_eq!(config.linear_num_key_heads, 16);
    assert_eq!(config.linear_num_value_heads(), 32);

    for module in ["in_proj_b", "in_proj_a"] {
        assert_eq!(
            qwen35_projection_shape(&config, 0, module),
            Ok(LoraProjectionShape {
                d_in: config.hidden_size,
                d_out: 32,
            }),
            "{module} must use the asymmetric value-head count"
        );
    }
}

#[test]
fn qwen35_projection_shape_accepts_mlp_modules_on_both_layer_types() {
    let config = Qwen35Config::qwen35_0_8b();
    let cases = [
        (
            "gate_proj",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 3584,
            },
        ),
        (
            "up_proj",
            LoraProjectionShape {
                d_in: 1024,
                d_out: 3584,
            },
        ),
        (
            "down_proj",
            LoraProjectionShape {
                d_in: 3584,
                d_out: 1024,
            },
        ),
    ];

    for layer_idx in [0, 3] {
        for (module, expected) in cases {
            assert_eq!(
                qwen35_projection_shape(&config, layer_idx, module),
                Ok(expected),
                "unexpected shape for {module} on layer {layer_idx}"
            );
        }
    }
}

#[test]
fn qwen35_projection_shape_rejects_wrong_layer_types() {
    let config = Qwen35Config::qwen35_0_8b();

    assert_eq!(
        qwen35_projection_shape(&config, 0, "q_proj"),
        Err("module 'q_proj' is a full-attention projection but layer 0 is GDN".to_string())
    );
    assert_eq!(
        qwen35_projection_shape(&config, 3, "in_proj_qkv"),
        Err("module 'in_proj_qkv' is a GDN projection but layer 3 is full-attention".to_string())
    );
}

#[test]
fn qwen35_projection_shape_rejects_unknown_and_out_of_range_targets() {
    let config = Qwen35Config::qwen35_0_8b();

    assert_eq!(
        qwen35_projection_shape(&config, 3, "q_porj"),
        Err("unknown LoRA module 'q_porj'".to_string())
    );
    assert_eq!(
        qwen35_projection_shape(&config, config.num_hidden_layers, "q_proj"),
        Err("layer 24 is out of range for Qwen3.5 model with 24 layers".to_string())
    );
}

#[test]
fn row_helper_is_the_same_function_item() {
    fn same_function<F>(_: F, _: F) {}
    same_function(
        super::apply_lora_rows,
        lattice_compute::lora_hook::apply_lora_rows,
    );
}

#[test]
fn qwen_geometry_matches_projection_helper_for_both_layer_kinds_and_errors() {
    let modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query",
        "unknown",
    ];
    for config in [Qwen35Config::qwen35_0_8b(), Qwen35Config::qwen36_35b_a3b()] {
        let geometry = Qwen35LoraGeometry::new(&config);
        assert_eq!(geometry.num_layers(), config.num_hidden_layers);
        let full = (0..config.num_hidden_layers)
            .find(|&layer| config.is_full_attention(layer))
            .unwrap();
        let linear = (0..config.num_hidden_layers)
            .find(|&layer| !config.is_full_attention(layer))
            .unwrap();
        for layer in [full, linear, config.num_hidden_layers] {
            for module in modules {
                assert_eq!(
                    geometry.projection_shape(layer, module),
                    qwen35_projection_shape(&config, layer, module),
                    "layer {layer}, module {module}"
                );
            }
        }
    }
}
