use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

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

struct RowSensitiveHook {
    calls: AtomicUsize,
}

impl LoraHook for RowSensitiveHook {
    fn apply(&self, _layer_idx: usize, _module: &str, x: &[f32], output: &mut [f32]) {
        assert_eq!(x.len(), 2);
        assert_eq!(output.len(), 3);
        self.calls.fetch_add(1, Ordering::Relaxed);
        output.fill(x[0]);
    }
}

#[test]
fn applies_lora_to_each_flattened_token_row() {
    let hook = RowSensitiveHook {
        calls: AtomicUsize::new(0),
    };
    let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
    let mut output = [0.0; 9];

    apply_lora_rows(&hook, 0, "projection", &input, &mut output, 2, 3);

    assert_eq!(hook.calls.load(Ordering::Relaxed), 3);
    assert_eq!(output, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
}

struct InactiveHook {
    calls: AtomicUsize,
}

impl LoraHook for InactiveHook {
    fn apply(&self, _layer_idx: usize, _module: &str, _x: &[f32], _output: &mut [f32]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn is_active(&self, _layer_idx: usize, _module: &str) -> bool {
        false
    }
}

/// A hook that reports `is_active == false` for a projection must never
/// have `apply` dispatched for any row of that projection: `apply_lora_rows`
/// checks activity once, before the per-row loop, not per row.
#[test]
fn skips_the_per_row_loop_entirely_when_the_hook_is_inactive() {
    let hook = InactiveHook {
        calls: AtomicUsize::new(0),
    };
    let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
    let mut output = [7.0; 9];

    apply_lora_rows(&hook, 0, "projection", &input, &mut output, 2, 3);

    assert_eq!(
        hook.calls.load(Ordering::Relaxed),
        0,
        "apply must not be called for any row when is_active is false"
    );
    assert_eq!(output, [7.0; 9], "output must be untouched");
}

/// `NoopLoraHook` (the default when no adapter is loaded) must report
/// itself inactive for every projection, so the default (no-adapter)
/// path stays off the per-row virtual-dispatch cost.
#[test]
fn noop_hook_reports_inactive_for_any_projection() {
    let hook = NoopLoraHook;
    assert!(!hook.is_active(0, "query"));
    assert!(!hook.is_active(41, "ffn_output"));
}
